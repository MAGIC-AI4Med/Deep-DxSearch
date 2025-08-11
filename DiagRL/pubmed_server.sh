export CUDA_VISIBLE_DEVICES=0
index_file=./your/path/to/pubmed/bm25luceneindex
corpus_file=./your/path/to/pubmed/corpus/pubmed.jsonl
retriever=intfloat/e5-base-v2
retriever_name=bm25
python ./your/path/to/pubmed_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever \
                                            --retriever_name $retriever_name \
                                            --faiss_gpu

