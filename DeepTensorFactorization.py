class DeepTensorFactorization(torch.nn.Module):
    def __init__(self, 
                 cell_types, 
                 compounds, 
                 genes, 
                 n_cell_type_factors: int=4, 
                 n_compounds_factors: int=16, 
                 n_gene_factors: int=128,
                 n_hiddens: int=2048,
                 dropout: float=0.1):
        super().__init__()

        self.cell_types = cell_types
        self.compounds = compounds
        self.genes = genes

        self.n_cell_types = len(cell_types)
        self.n_compounds = len(compounds)
        self.n_genes = len(genes)

        self.n_cell_type_factors = n_cell_type_factors
        self.n_compounds_factors = n_compounds_factors
        self.n_gene_factors = n_gene_factors

        self.cell_type_embedding = torch.nn.Embedding(self.n_cell_types, self.n_cell_type_factors)
        self.compound_embedding = torch.nn.Embedding(self.n_compounds, self.n_compounds_factors)
        self.gene_embedding = torch.nn.Embedding(self.n_genes, self.n_gene_factors)

        self.n_hiddens = n_hiddens
        self.dropout = dropout
        self.n_factors = n_cell_type_factors + n_compounds_factors + n_gene_factors

        self.model = nn.Sequential(nn.Linear(self.n_factors, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, self.n_hiddens),
                                   nn.BatchNorm1d(self.n_hiddens),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.n_hiddens, 1))

    def forward(self, cell_type_indices, compound_indices, gene_indices):
        cell_type_vec = self.cell_type_embedding(cell_type_indices)
        compound_vec = self.compound_embedding(compound_indices)
        gene_vec = self.gene_embedding(gene_indices)

        x = torch.concat([cell_type_vec, compound_vec, gene_vec], dim=1)
        x = self.model(x)

        return x
