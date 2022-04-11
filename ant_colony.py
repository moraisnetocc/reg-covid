'''
Localizar uma formiga em um vértice do grafo G=(V,E)
Para t←1 até o número de iterações (ou colônias)
 Para k←1 até m
 Enquanto a formiga k não construir o caminho Sk
 Selecione o próximo vértice pela regra Pij
k
 Fim Enquanto
 Calcule a distância Lk do caminho Sk
 Se Lk<L* então
 S*← Sk, L*←Lk
 Fim Se
 Fim Para
 Atualize os feromônios
Fim Para
Retornar S*
'''


