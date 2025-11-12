import pandas as pd

def prepare_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    if 'book_id' not in books.columns:
        if 'id' in books.columns:
            books = books.rename(columns={'id': 'book_id'})
    for c in ['title','authors']:
        if c not in books.columns:
            books[c] = ''
    books['title'] = books['title'].fillna('').astype(str)
    books['authors'] = books['authors'].fillna('').astype(str)
    books['combined'] = (books['title'] + ' ' + books['authors']).str.lower()
    books['book_id'] = books['book_id'].astype(str)
    return books
