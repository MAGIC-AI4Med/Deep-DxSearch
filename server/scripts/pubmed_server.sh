export CUDA_VISIBLE_DEVICES=0
index_file=/mnt/vision_user/zhengqiaoyu/DiagRL/server/pubmed/bm25luceneindex
corpus_file=/mnt/vision_user/zhengqiaoyu/DiagRL/server/pubmed/corpus/pubmed.jsonl
retriever=intfloat/e5-base-v2
retriever_name=bm25
/mnt/vision_user/yibinyan/miniconda3/envs/retriever/bin/python3 /mnt/vision_user/zhengqiaoyu/DiagRL/server/server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_model $retriever \
                                            --retriever_name $retriever_name \
                                            --faiss_gpu --port 8000 \

