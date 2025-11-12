import pandas as pd
def prepare_books(books: pd.DataFrame, title_col='title', author_col='authors', id_col='book_id', rating_col=None):
    books = books.copy()
    # standardize columns
    if title_col not in books.columns:
        # try lowercase match
        for c in books.columns:
            if c.lower() == title_col:
                title_col = c
    if author_col not in books.columns:
        for c in books.columns:
            if 'author' in c.lower():
                author_col = c
                break
    if id_col not in books.columns:
        for c in books.columns:
            if 'id' in c.lower() or 'isbn' in c.lower():
                id_col = c
                break
    # ensure existence
    if title_col not in books.columns:
        books['title'] = books.iloc[:,0].astype(str)
        title_col = 'title'
    if author_col not in books.columns:
        books['authors'] = ''
        author_col = 'authors'
    if id_col not in books.columns:
        books['book_id'] = books.index.astype(str)
        id_col = 'book_id'
    # fillna and types
    books[title_col] = books[title_col].fillna('').astype(str)
    books[author_col] = books[author_col].fillna('').astype(str)
    books[id_col] = books[id_col].astype(str)
    # combined text
    books['combined'] = (books[title_col] + ' ' + books[author_col]).str.lower()
    # if rating_col provided and exists, keep numeric else None
    if rating_col and rating_col in books.columns:
        try:
            books['__rating__'] = pd.to_numeric(books[rating_col], errors='coerce')
        except:
            books['__rating__'] = None
    else:
        books['__rating__'] = None
    return books.rename(columns={title_col:'title', author_col:'authors', id_col:'book_id'})
