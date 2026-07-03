"""Headless executor for tabfm_vs_catboost.ipynb (long-running; use nohup)."""
import nbformat
from nbclient import NotebookClient

PATH = 'tabfm_vs_catboost.ipynb'
nb = nbformat.read(PATH, as_version=4)
client = NotebookClient(nb, timeout=None, kernel_name='python3')
client.execute()
nbformat.write(nb, PATH)
print('notebook executed OK')
