def top_popular(books_df, topn=10):
    # expects '__rating__' if available, else try 'ratings_count' or fallback to alphabetical
    if '__rating__' in books_df.columns and books_df['__rating__'].notna().any():
        df = books_df.dropna(subset=['__rating__']).sort_values('__rating__', ascending=False)
    elif 'ratings_count' in books_df.columns:
        df = books_df.sort_values('ratings_count', ascending=False)
    else:
        df = books_df.sort_values('title')
    return df.head(topn)[['book_id','title','authors','__rating__']].to_dict(orient='records')
