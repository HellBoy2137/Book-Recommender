import pandas as pd

def prepare_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    # normalize columns
    if 'book_id' not in books.columns:
        for c in ['id','bookId']:
            if c in books.columns:
                books.rename(columns={c:'book_id'}, inplace=True)
    for c in ['title','authors']:
        if c not in books.columns:
            books[c] = ''
    books['title'] = books['title'].fillna('').astype(str)
    books['authors'] = books['authors'].fillna('').astype(str)
    books['combined'] = (books['title'] + ' ' + books['authors']).str.lower()
    books['book_id'] = books['book_id'].astype(str)
    return books
