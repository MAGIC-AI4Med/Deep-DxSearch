python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./your/path/to/wikipedia/corpus \
  --index ./your/path/to/wikipedia/bm25luceneindex \
  --generator DefaultLuceneDocumentGenerator \
  --threads 8 \
  --storePositions --storeDocvectors --storeRaw