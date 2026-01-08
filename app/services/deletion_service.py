from app.services.metadata_service import (
    get_vector_keys_by_s3_key,
    mark_document_deleted,
    delete_document_metadata,
)
from app.services.vector_service import delete_vectors_by_keys


def delete_embeddings_and_metadata(s3_key: str, delete_metadata: bool) -> tuple[int, bool, bool]:
    vector_keys = get_vector_keys_by_s3_key(s3_key)
    if vector_keys:
        delete_vectors_by_keys(vector_keys)

    meta_deleted = False
    meta_marked = False
    if delete_metadata:
        meta_deleted = delete_document_metadata(s3_key)
    else:
        meta_marked = mark_document_deleted(s3_key)

    return (len(vector_keys), bool(meta_deleted), bool(meta_marked))


