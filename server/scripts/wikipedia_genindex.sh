python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /mnt/vision_user/zhengqiaoyu/DiagRL/server/wikipedia/corpus \
  --index /mnt/vision_user/zhengqiaoyu/DiagRL/server/wikipedia/bm25luceneindex \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw

