export CUDA_VISIBLE_DEVICES=0
index_file=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangya-24047/raredata/wikipedia/bm25luceneindex
corpus_file=/inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangya-24047/raredata/wikipedia/corpus/wikipedia.jsonl
retriever=intfloat/e5-base-v2
retriever_name=bm25
python /inspire/ssd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhangya-24047/retrieve_server/wiki_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever \
                                            --retriever_name $retriever_name \
                                            --faiss_gpu