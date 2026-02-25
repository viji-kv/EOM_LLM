import os
import json
from typing import Any, List, Dict, Optional

from langchain_community.embeddings import OpenAIEmbeddings
from supabase.client import Client, create_client
from supabase.lib.client_options import ClientOptions

# from ai_assistant.vector_store.supabase_db import CustomSupabaseVectorStore
from supabase_utils.supabase_db import CustomSupabaseVectorStore


def initialize_supabase() -> Client:
    """
    Initialize Supabase client with proper authentication.
    Returns:
        Supabase client instance.
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


def get_workspaces(supabase: Client) -> List[Dict[str, Any]]:
    """
    Fetch all workspaces from the database.
    """
    response = supabase.table("workspaces").select("*").execute()
    if response is None or response == []:
        print(f"Error fetching workspaces: {response.error.message}")
        return []
    return response.data or []


def get_brains_per_workspace(
    supabase: Client, workspace_id: Any
) -> List[Dict[str, Any]]:
    """
    Fetch detailed brain information associated with a specific workspace.
    """
    # Step 1: Fetch brain_ids from workspaces_brains table
    response = (
        supabase.table("workspaces_brains")
        .select("brain_id")
        .eq("workspace_id", workspace_id)
        .execute()
    )
    if response is None or response == []:
        print(
            f"Error fetching brain IDs for workspace {workspace_id}: {response.error.message}"
        )
        return []

    brain_ids = [record["brain_id"] for record in response.data]

    if not brain_ids:
        return []

    # Step 2: Fetch brain details from brains table using brain_ids
    brains_response = (
        supabase.table("brains").select("*").in_("brain_id", brain_ids).execute()
    )
    if brains_response is None or brains_response == []:
        print(
            f"Error fetching brains for workspace {workspace_id}: {brains_response.error.message}"
        )
        return []

    return brains_response.data or []


def get_documents_per_brain(supabase: Client, brain_id: Any) -> List[Dict[str, Any]]:
    """
    Fetch documents associated with a specific brain.
    """
    response = (
        supabase.table("knowledge").select("*").eq("brain_id", brain_id).execute()
    )
    if response is None or response == []:
        print(
            f"Error fetching documents for brain {brain_id}: {response.error.message}"
        )
        return []
    return response.data or []


def filter_unique_items(
    items: List[Dict[str, Any]], unique_key: str
) -> List[Dict[str, Any]]:
    """
    Filter a list of dictionaries to only include unique items based on a specified key.
    """
    seen = set()
    unique_items = []
    duplicates_found = False
    for item in items:
        key = item.get(unique_key)
        if key not in seen:
            unique_items.append(item)
            seen.add(key)
        else:
            duplicates_found = True
    if duplicates_found:
        print(
            f"Note: Duplicate entries found based on '{unique_key}' have been removed."
        )
    return unique_items


def display_list(items: List[Dict[str, Any]], display_key: str) -> None:
    """
    Display a numbered list of items based on a specific key.
    """
    for idx, item in enumerate(items, start=1):
        print(f"{idx}. {item.get(display_key, 'N/A')}")


def get_selection(num_items: int) -> Optional[List[int]]:
    """
    Prompt the user to select items by entering numbers separated by commas.
    Returns:
        A list of selected indices or None if the user chooses to stop.
    """
    while True:
        selection = input(
            "Enter the numbers of your selections, separated by commas (or '0' to stop): "
        )
        if selection.strip() == "0":
            return None  # User chose to stop
        try:
            selected_indices = [int(num.strip()) for num in selection.split(",")]
            # Validate the indices
            if all(1 <= idx <= num_items for idx in selected_indices):
                return list(set(selected_indices))  # Remove duplicates
            else:
                print(
                    f"Please enter numbers between 1 and {num_items}, or '0' to stop."
                )
        except ValueError:
            print(
                "Invalid input. Please enter numbers separated by commas, or '0' to stop."
            )


def select():
    supabase = initialize_supabase()
    selections = {"workspaces": [], "brains": [], "documents": []}

    # Step 1: Fetch and Display Workspaces
    print("\nFetching workspaces...")
    workspaces = get_workspaces(supabase)
    if not workspaces:
        print("No workspaces found.")
        return selections

    # Filter unique workspaces based on 'id'
    workspaces = filter_unique_items(
        workspaces, "workspace_id"
    )  # Assuming 'id' is unique
    if not workspaces:
        print("No unique workspaces available after removing duplicates.")
        return selections

    print("\nAvailable Workspaces:")
    display_list(workspaces, "name")  # Assuming each workspace has a 'name' field
    selected_workspace_indices = get_selection(len(workspaces))
    if selected_workspace_indices is None:
        print("Selection stopped. Returning selected workspaces (none).")
        return selections

    selected_workspaces = [workspaces[idx - 1] for idx in selected_workspace_indices]
    selections["workspaces"] = selected_workspaces
    selected_workspace_ids = [ws["workspace_id"] for ws in selected_workspaces]

    # Step 2: Fetch and Display Brains for Selected Workspaces
    all_brains = []
    for ws_id in selected_workspace_ids:
        brains = get_brains_per_workspace(supabase, ws_id)
        if brains:
            all_brains.extend(brains)

    if not all_brains:
        print("No brains found for the selected workspaces.")
        return selections

    # Filter unique brains based on 'id'
    all_brains = filter_unique_items(all_brains, "brain_id")  # Assuming 'id' is unique
    if not all_brains:
        print("No unique brains available after removing duplicates.")
        return selections

    print("\nAvailable Brains:")
    display_list(all_brains, "name")  # Assuming each brain has a 'name' field
    selected_brain_indices = get_selection(len(all_brains))
    if selected_brain_indices is None:
        print("Selection stopped. Returning selected workspaces and brains.")
        return selections

    selected_brains = [all_brains[idx - 1] for idx in selected_brain_indices]
    selections["brains"] = selected_brains
    selected_brain_ids = [brain["brain_id"] for brain in selected_brains]

    # Step 3: Fetch and Display Documents for Selected Brains
    all_documents = []
    for brain_id in selected_brain_ids:
        documents = get_documents_per_brain(supabase, brain_id)
        if documents:
            all_documents.extend(documents)

    if not all_documents:
        print("No documents found for the selected brains.")
        return selections

    # Filter unique documents based on 'id'
    all_documents = filter_unique_items(all_documents, "id")  # Assuming 'id' is unique
    if not all_documents:
        print("No unique documents available after removing duplicates.")
        return selections

    print("\nAvailable Documents:")
    display_list(
        all_documents, "file_name"
    )  # Assuming each document has a 'title' field
    selected_document_indices = get_selection(len(all_documents))
    if selected_document_indices is None:
        print(
            "Selection stopped. Returning selected workspaces, brains, and documents."
        )
        return selections

    selected_documents = [all_documents[idx - 1] for idx in selected_document_indices]
    selections["documents"] = selected_documents

    # Step 4: Display Selected Documents
    print("\nYou have selected the following documents:")
    for doc in selected_documents:
        print(f"- {doc.get('file_name', 'N/A')} - {doc.get('id', 'N/A')}")

    # Optionally, you can process the selected documents further here
    # For example, display their content or perform other operations

    return selections


# EXTENDED: Select all docs in a brain
def select_brain_from_workspace():
    supabase = initialize_supabase()
    selections = {"workspaces": [], "brains": []}

    # Step 1: Fetch and Display Workspaces
    print("\nFetching workspaces...")
    workspaces = get_workspaces(supabase)
    if not workspaces:
        print("No workspaces found.")
        return selections

    # Filter unique workspaces based on 'id'
    workspaces = filter_unique_items(
        workspaces, "workspace_id"
    )  # Assuming 'id' is unique
    if not workspaces:
        print("No unique workspaces available after removing duplicates.")
        return selections

    print("\nAvailable Workspaces:")
    display_list(workspaces, "name")  # Assuming each workspace has a 'name' field
    selected_workspace_indices = get_selection(len(workspaces))
    if selected_workspace_indices is None:
        print("Selection stopped. Returning selected workspaces (none).")
        return selections

    selected_workspaces = [workspaces[idx - 1] for idx in selected_workspace_indices]
    selections["workspaces"] = selected_workspaces
    selected_workspace_ids = [ws["workspace_id"] for ws in selected_workspaces]

    # Step 2: Fetch and Display Brains for Selected Workspaces
    all_brains = []
    for ws_id in selected_workspace_ids:
        brains = get_brains_per_workspace(supabase, ws_id)
        if brains:
            all_brains.extend(brains)

    if not all_brains:
        print("No brains found for the selected workspaces.")
        return selections

    # Filter unique brains based on 'id'
    all_brains = filter_unique_items(all_brains, "brain_id")  # Assuming 'id' is unique
    if not all_brains:
        print("No unique brains available after removing duplicates.")
        return selections

    print("\nAvailable Brains:")
    display_list(all_brains, "name")  # Assuming each brain has a 'name' field
    selected_brain_indices = get_selection(len(all_brains))
    if selected_brain_indices is None:
        print("Selection stopped. Returning selected workspaces and brains.")
        return selections

    selected_brains = [all_brains[idx - 1] for idx in selected_brain_indices]
    selections["brains"] = selected_brains

    selected_brain = selected_brains[0]  # First brain
    return selected_brain  # Single brain dict!


if __name__ == "__main__":
    selections = select()
    print("\nFinal Selections:")
    print(json.dumps(selections, indent=4))
