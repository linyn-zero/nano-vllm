from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

"""
BlockManager 块管理器
  属于 Sequence 的资源抽象底层
  本质是 cache for token_ids，位于 CPU。
  实现了前缀共享机制。
  block 通过 model_runner 的 slot_mapping 绑定 model(Attention module) 的 kv cache
"""


class Block:
    """
    块，支持前缀共享
    """
    def __init__(self, block_id):
        self.block_id = block_id  # 初始化
        self.ref_count = 0  # 分配缓存块资源
        self.hash = -1  # 写入缓存
        self.token_ids = []  # 写入缓存

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """
    1. 资源池策略：以 Resource pool 方式管理缓存块资源
    2. 哈希策略：使用前一个 block 的哈希值加当前 block 的内容（即 token_ids）来计算当前 block 的哈希值，使用哈希值进行 block 的区分与共享。
    3. 缓存策略：直接映射。缓存块回收后不清空，再分配时直接复写（或恰好复用）。缓存块解偶了分配与写入，有一个新 token 时分配，token 能填满 block 时写入。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
                
        # --- 分配缓存块资源 ---
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()  # 辅助 block.ref_count 计数
        
        # --- 写入缓存（对外可见） ---
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]  # cache block （直接映射）
        self.hash_to_block_id: dict[int, int] = dict()  # 找缓存地址（资源池策略时序分配资源，因此需要建立 token_ids 到 cache block 的映射表）


    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        prefix 是前一个 block 的 hash
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # prefill
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks  # 没考虑可共享的缓存 

    def allocate(self, seq: Sequence):
        assert not seq.block_table  # 初始化 seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1  # 能填满的 block 才有哈希
            block_id = self.hash_to_block_id.get(h, -1)  # 旧哈希还是新哈希还是没哈希
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:  # block 不能填满或新哈希或旧的无效的哈希：cache未命中
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:  
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:  # cache 命中但实际上该 block 已回收，恰好复用
                    block = self._allocate_block(block_id)  # 但实际上还是会清空 block 内容
            if h != -1:  # 能填满 block 才写入 cache block
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)  

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # decode (append cache block BEFORE decode the new token)
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)  # 缓存块分配时机

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:  # allocate new cache block
            assert last_block.hash != -1  # last block must be full
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:  # last block must be full, write into it
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:  # the new block is not full
            assert last_block.hash == -1
