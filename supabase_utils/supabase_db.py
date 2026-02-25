import codecs
import json
import uuid

# Load environment variables
from typing import Any, List, Optional, Tuple, Dict, Union
from supabase.client import Client, create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from supabase.client import Client
from supabase.lib.client_options import ClientOptions
import warnings
from langchain_community.embeddings import OpenAIEmbeddings


def initialize_supabase():
    """
    Initialize Supabase client with proper authentication
    Returns: Supabase client instance
    """

    SUPABASE_URL = "http://188.166.5.51:54321"
    SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise ValueError("Missing required Supabase credentials")

    return create_client(
        SUPABASE_URL,
        SUPABASE_SERVICE_KEY,
        options=ClientOptions(
            postgrest_client_timeout=1000,
            storage_client_timeout=1000,
            schema="public",
        ),
    )


def initialize_vector_store():
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize Supabase client
    supabase_client = initialize_supabase()

    # Create custom vector store
    vector_store = CustomSupabaseVectorStore(
        embedding=embeddings,
        client=supabase_client,
        table_name="vectors",
        query_name="match_documents",
    )
    return vector_store


def get_workspace_id(supabase, name):

    return supabase.table("workspaces").select("*").eq("name", name).execute().data


def get_brains_per_workspace(supabase, workspace_id):
    return (
        supabase.table("workspaces_brains")
        .select("*")
        .eq("workspace_id", workspace_id)
        .execute()
        .data
    )


def get_documents_per_brain(supabase, brain_id):
    return (
        supabase.table("knowledge").select("*").eq("brain_id", brain_id).execute().data
    )


def get_document_data(supabase, document_id, batch_size=50):
    vector_ids = (
        supabase.table("brains_vectors")
        .select("vector_id")
        .eq("knowledge_id", document_id)
        .order("order", desc=False)
        .execute()
        .data
    )

    texts = []
    for i in range(0, len(vector_ids), batch_size):
        batch = vector_ids[i : i + batch_size]
        batch_results = (
            supabase.table("vectors")
            .select("content")
            .in_("id", [vector_id["vector_id"] for vector_id in batch])
            .execute()
            .data
        )
        texts.extend(batch_results)

    return "\n".join([text["content"] for text in texts])


def decode_string(s):
    """
    Replaces literal "\\n" with actual newlines and decodes Unicode escape sequences.

    :param s: Original string with escaped characters
    :return: Decoded string with actual newlines and Unicode characters
    """
    # Replace literal "\\n" with "\n"
    s = s.replace("\\n", "\n")
    # Decode Unicode escape sequences (e.g., "\\u2019" to "’")
    s = codecs.decode(s, "unicode_escape")
    return s


def get_vectors_by_knowledge_ids(supabase, knowledge_ids):
    """
    Retrieves vectors associated with one or multiple knowledge IDs in a single query.

    Args:
        knowledge_ids (str or List[str]): A single knowledge ID or a list of knowledge IDs.

    Returns:
        Dict[str, List[dict]]: A dictionary where each key is a knowledge ID and the value is a list of vector chunks.
    """
    # Ensure knowledge_ids is a list
    if isinstance(knowledge_ids, str):
        knowledge_ids = [knowledge_ids]
    elif isinstance(knowledge_ids, list):
        # Optionally, validate that all items in the list are strings
        if not all(isinstance(kid, str) for kid in knowledge_ids):
            raise ValueError("All knowledge_ids must be strings.")
    else:
        raise TypeError("knowledge_ids must be a string or a list of strings.")

    try:
        response = (
            supabase.from_("brains_vectors")
            .select("knowledge_id, vector_id, vectors(metadata, content)")
            .in_("knowledge_id", knowledge_ids)
            .order("knowledge_id", desc=False)  # Primary ordering by knowledge_id
            .order("order", desc=False)  # Secondary ordering by 'order' field
            .execute()
        )
    except Exception as e:
        # Handle unexpected exceptions
        print(f"Unexpected error during query execution: {e}")
        return {}

    # Initialize a dictionary to hold results per knowledge_id
    chunks_dict = {kid: [] for kid in knowledge_ids}

    for item in response.data:
        knowledge_id = item.get("knowledge_id")
        vector_id = item.get("vector_id")
        vectors = item.get("vectors", {})
        content = vectors.get("content", "")
        metadata = vectors.get("metadata", {})

        if knowledge_id and vector_id and content:
            # Append the chunk to the corresponding knowledge_id
            if knowledge_id in chunks_dict:
                chunks_dict[knowledge_id].append(decode_string(content))
            else:
                # This case should not occur if knowledge_ids are properly provided
                chunks_dict[knowledge_id] = [decode_string(content)]
        else:
            # Handle incomplete data
            print(
                f"Incomplete data for vector_id: {vector_id}, knowledge_id: {knowledge_id}"
            )

    return {key: "\n".join(value_list) for key, value_list in chunks_dict.items()}


def get_summary_by_knowledge_ids(supabase, knowledge_ids):
    """
    Retrieves vectors associated with one or multiple knowledge IDs in a single query.

    Args:
        knowledge_ids (str or List[str]): A single knowledge ID or a list of knowledge IDs.

    Returns:
        Dict[str, List[dict]]: A dictionary where each key is a knowledge ID and the value is a list of vector chunks.
    """
    # Ensure knowledge_ids is a list
    if isinstance(knowledge_ids, str):
        knowledge_ids = [knowledge_ids]
    elif isinstance(knowledge_ids, list):
        # Optionally, validate that all items in the list are strings
        if not all(isinstance(kid, str) for kid in knowledge_ids):
            raise ValueError("All knowledge_ids must be strings.")
    else:
        raise TypeError("knowledge_ids must be a string or a list of strings.")

    try:
        response = (
            supabase.from_("knowledge")
            .select("id, file_name, url, summary, summary_embedding")
            .in_("id", knowledge_ids)
            .execute()
        )
    except Exception as e:
        # Handle unexpected exceptions
        print(f"Unexpected error during query execution: {e}")
        return {}

    response_data = []

    for item in response.data:
        knowledge_id = item.get("id")
        embedding = item.get("summary_embedding")
        description = item.get("summary", "")
        url = item.get("url")
        response_data.append(
            {
                "knowledge_id": knowledge_id,
                "url": url,
                "description": description,
                "embedding": eval(embedding) if embedding else None,
                "file_name": item.get("file_name"),
            }
        )

    return response_data


class CustomSupabaseVectorStore(SupabaseVectorStore):
    """A custom vector store that uses the match_vectors table instead of the vectors table."""

    number_docs: int = 35
    max_input: int = 2000

    def __init__(
        self,
        client: Client,
        embedding: Embeddings,
        table_name: str,
        number_docs: int = 35,
        max_input: int = 2000,
        query_name: Union[str, None] = None,
    ):
        super().__init__(client, embedding, table_name)
        self.supabase = client
        self.query_name = query_name or "match_documents"
        self.number_docs = number_docs
        self.max_input = max_input

    def similarity_search_by_vector_with_relevance_scores(
        self,
        query: List[float],
        k: int = 1000,
        filter: Optional[Dict[str, Any]] = None,
        postgrest_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        knowledge_ids = None
        brain_id = None
        if filter is not None:
            filter_dict = json.loads(filter)
            if "knowledge_ids" in filter_dict:
                knowledge_ids = filter_dict.pop("knowledge_ids")
            if "p_brain_id" in filter_dict:
                brain_id = filter_dict.pop("p_brain_id")
            if filter_dict:  # Check if filter_dict is not empty
                match_documents_params = self.match_args(query, str(filter_dict))
            else:
                match_documents_params = self.match_args(query, None)

        else:
            match_documents_params = self.match_args(query, filter)

        if brain_id and brain_id != "" and brain_id != "none":
            match_documents_params["p_brain_id"] = brain_id
        if knowledge_ids is not None and len(knowledge_ids) > 0:
            match_documents_params["knowledge_ids"] = knowledge_ids

        query_builder = self._client.rpc(self.query_name, match_documents_params)
        if postgrest_filter:
            query_builder.params = query_builder.params.set(
                "and", f"({postgrest_filter})"
            )

        query_builder.params = query_builder.params.set("limit", k)

        res = query_builder.execute()

        match_result = [
            (
                Document(
                    metadata={
                        **search.get("metadata", {}),
                        "id": search.get("id", ""),
                        "similarity": search.get("similarity", 0.0),
                        "knowledge_id": search.get("knowledge_id", None),
                    },
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        if score_threshold is not None:
            match_result = [
                (doc, similarity)
                for doc, similarity in match_result
                if similarity >= score_threshold
            ]
            if len(match_result) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )

        return match_result


def main():
    from ai_assistant.vector_store.select_data import (
        get_selection,
        display_list,
        filter_unique_items,
        get_workspaces,
    )

    supabase = initialize_supabase()

    get_document_data(supabase, "e846d975-9784-43ef-bf20-7328b5fe301c")
    # Step 1: Fetch and Display Workspaces
    print("Fetching workspaces...")
    workspaces = get_workspaces(supabase)
    if not workspaces:
        print("No workspaces found.")
        return

    # Filter unique workspaces based on 'id' and 'name'
    workspaces = filter_unique_items(
        workspaces, "workspace_id"
    )  # Assuming 'id' is unique
    if not workspaces:
        print("No unique workspaces available after removing duplicates.")
        return

    print("\nAvailable Workspaces:")
    display_list(workspaces, "name")  # Assuming each workspace has a 'name' field
    selected_workspace_indices = get_selection(len(workspaces))
    selected_workspaces = [workspaces[idx - 1] for idx in selected_workspace_indices]
    selected_workspace_ids = [ws["workspace_id"] for ws in selected_workspaces]

    # Step 2: Fetch and Display Brains for Selected Workspaces
    all_brains = []
    for ws_id in selected_workspace_ids:
        brains = get_brains_per_workspace(supabase, ws_id)
        if brains:
            all_brains.extend(brains)

    if not all_brains:
        print("No brains found for the selected workspaces.")
        return

    # Filter unique brains based on 'id' and 'name'
    all_brains = filter_unique_items(all_brains, "brain_id")  # Assuming 'id' is unique
    if not all_brains:
        print("No unique brains available after removing duplicates.")
        return

    print("\nAvailable Brains:")
    display_list(all_brains, "name")  # Assuming each brain has a 'name' field
    selected_brain_indices = get_selection(len(all_brains))
    selected_brains = [all_brains[idx - 1] for idx in selected_brain_indices]
    selected_brain_ids = [brain["brain_id"] for brain in selected_brains]

    # Step 3: Fetch and Display Documents for Selected Brains
    all_documents = []
    for brain_id in selected_brain_ids:
        documents = get_documents_per_brain(supabase, brain_id)
        if documents:
            all_documents.extend(documents)

    if not all_documents:
        print("No documents found for the selected brains.")
        return

    # Filter unique documents based on 'id' and 'title'
    all_documents = filter_unique_items(all_documents, "id")  # Assuming 'id' is unique
    if not all_documents:
        print("No unique documents available after removing duplicates.")
        return

    print("\nAvailable Documents:")
    display_list(
        all_documents, "file_name"
    )  # Assuming each document has a 'title' field
    selected_document_indices = get_selection(len(all_documents))
    selected_documents = [all_documents[idx - 1] for idx in selected_document_indices]

    # Step 4: Display Selected Documents
    print("\nYou have selected the following documents:")
    for doc in selected_documents:
        print(f"- {doc.get('file_name', 'N/A')} - {doc.get('id', 'N/A')}")

    print(get_document_data(supabase, [doc["id"] for doc in selected_documents]))
    # Optionally, process the selected documents further here
    # For example, display their content or perform other operations


if __name__ == "__main__":
    main()
