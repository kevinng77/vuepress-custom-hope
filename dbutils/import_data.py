from utils.doc_utils import convert_files_to_dicts

from esutils import EsUtils
from retriever.dense import DensePassageRetriever 


def main():
    document_store = EsUtils()
    dicts = convert_files_to_dicts(
        dir_path="./data", 
        split_paragraphs=True, 
        encoding="utf-8",
        
    )
    print(set([d['meta']["name"] for d in dicts]))
    print(dicts[:3])
    document_store.write_documents(dicts)
    # 语义索引模型
    retriever = DensePassageRetriever(
        document_store=document_store,
        query_embedding_model="rocketqa-zh-nano-query-encoder",    # from_pretrain(query_embedding_model)
        passage_embedding_model="rocketqa-zh-nano-passage-encoder",
        params_path=None,
        output_emb_size=312 if "ernie" in ["ernie_search", "neural_search"] else None,
        share_parameters=False,
        max_seq_len_query=64,
        max_seq_len_passage=256,
        batch_size=16,
        use_gpu=True,
        embed_title=False,
    )

    # # 建立索引库
    document_store.update_embeddings(retriever)

if __name__ == "__main__":
    main()

