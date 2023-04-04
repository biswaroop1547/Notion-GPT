# Notion-GPT


```
pip install -r requirements.txt
```

## Usage

### Create vectorstore from your notion page.

1. Export notion page as Markdown (select to include subpages)
2. `git clone https://github.com/biswaroop1547/Notion-GPT.git`
3. `cd Notion-GPT`
4. `unzip <exported_file>.zip` and save it with directory name - `Notion_DB` inside `Notion-GPT` directory.

Now to create vectorstore run:
`./ingest.sh`

## To start the server
`make start`