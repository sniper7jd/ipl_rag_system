from langchain_community.document_loaders import WikipediaLoader


def fetch_team_data(teams, load_max_docs=1):
    all_docs = []
    for team in teams:
        loader = WikipediaLoader(query=team, load_max_docs=load_max_docs)
        all_docs.extend(loader.load())
    return all_docs
