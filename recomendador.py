import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def cargar_datos(ruta_ratings, ruta_movies):
    ratings = pd.read_csv(ruta_ratings)
    movies = pd.read_csv(ruta_movies)
    return ratings, movies

def crear_matriz_utilidad(ratings):
    return ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

def calcular_similitud(matriz_utilidad):
    return cosine_similarity(matriz_utilidad)

def recomendar(usuario_id, matriz_utilidad, similitudes, top_n=5):
    sim_usuarios = similitudes[usuario_id - 1]  # userId empieza desde 1
    productos_usuario = matriz_utilidad.loc[usuario_id]
    productos_no_vistos = productos_usuario[productos_usuario == 0].index

    recomendacion = pd.Series(dtype=float)

    for otro_usuario, score in enumerate(sim_usuarios):
        if otro_usuario + 1 == usuario_id:
            continue
        productos_otro = matriz_utilidad.loc[otro_usuario + 1]
        for producto in productos_no_vistos:
            if productos_otro[producto] > 0:
                recomendacion[producto] = recomendacion.get(producto, 0) + productos_otro[producto] * score

    return recomendacion.sort_values(ascending=False).head(top_n)
