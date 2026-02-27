python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input /mnt/vision_user/zhengqiaoyu/DiagRL/server/textbooks/corpus \
  --index /mnt/vision_user/zhengqiaoyu/DiagRL/server/textbooks/bm25luceneindex \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw

