# Reference: https://svds.com/jupyter-notebook-best-practices-for-data-science/
import os.path as osp
from subprocess import check_call


def convert_notebook_to_html(work_dir, fname):
    output_fname = fname + '.html'  # extension: .ipynb.html
    check_call(['jupyter', 'nbconvert', '--to', 'html', '--no-prompt',
                '--output', output_fname, fname], cwd=work_dir)


def convert_notebook_to_python(work_dir, fname):
    output_fname = fname + '.py'  # extension: .ipynb.py
    check_call(['jupyter', 'nbconvert', '--to', 'python', '--no-prompt',
                '--output', output_fname, fname], cwd=work_dir)


def convert_notebook(os_path):
    work_dir, fname = osp.split(os_path)
    # convert_notebook_to_html(work_dir, fname)
    convert_notebook_to_python(work_dir, fname)


def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts and .html files"""
    if model['type'] == 'notebook':
        convert_notebook(os_path)


def check_post_save():
    model = {"type": "notebook"}
    os_path = osp.abspath("my_notebook.ipynb")
    contents_manager = None
    post_save(model, os_path, contents_manager)


if not __name__ == '__main__':
    c.FileContentsManager.post_save_hook = post_save
else:
    check_post_save()
