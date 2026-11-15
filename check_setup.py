from helpdesk.ingest import embed

print("Embedding length:", len(embed(["hello world"])[0]))
