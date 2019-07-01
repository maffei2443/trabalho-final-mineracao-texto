- Não foi possível implementar o kmeans++ utilizando mais de uma thread.
Portanto, para fins de obter-se compações justas, foi utilizada apenas
uma thread. Note-se que isto é uma forte limitação da atual implementa-
ção do *kmeans++*.

- O código espera uma matriz esparsa de tipo scipy.sparse.csr_matrix.
Entretanto, há outros formatos de matrizes esparsas. Optou-se por consi-
derar que os dados virão no citado formato devido à esse ser o formato
padrão de saída do método *fit\_transform* da classe 
sklearn.feature_extraction.text.TfidfVectorizer

