# SimRes

Repositório de simulações de escoamento em reservatórios, com soluções analíticas e métodos de diferenças finitas 1D e 2D.

## Estrutura

- `src/analytical/`: soluções analíticas lineares e radiais.
- `src/mdf/`: métodos de diferenças finitas, modelos de poço, permeabilidade, falhas e velocidades.
- `src/core/`: métricas de erro e solvers lineares auxiliares.
- `notebooks/analytical/`: visualizações das soluções analíticas.
- `notebooks/numerical/`: simulações numéricas por geometria.
- `notebooks/validation/`: estudos de convergência e comparação de solvers.
- `data/`: dados de entrada, incluindo a matriz de permeabilidade.

## Ambiente

Crie o ambiente a partir de `environment.devenv.yml` (Conda/Devenv). O ambiente configura `PYTHONPATH=src`, necessário para os imports dos notebooks.

Abra o Jupyter a partir da raiz do repositório para que os caminhos relativos dos dados, como `data/permeabilidade.csv`, sejam encontrados. Os notebooks guardam gráficos e saídas executadas no próprio arquivo.

## Navegação sugerida

1. Explore primeiro os notebooks em `notebooks/analytical/`.
2. Compare os esquemas numéricos em `notebooks/numerical/linear_1d/` e os cenários 2D em `notebooks/numerical/reservoir_2d/`.
3. Consulte convergência e desempenho em `notebooks/validation/`.
