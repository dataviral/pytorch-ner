import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMNER(nn.Module):

    def __init__(
        self,
        n_classes,
        n_embeddings,
        embed_dims,
        n_lstm_layers,
        lstm_dims=256
    ):
        super(LSTMNER, self).__init__()

        """model hyperparameters"""
        self.n_classes = n_classes
        self.n_embeddings = n_embeddings
        self.embed_dims = embed_dims
        self.n_lstm_layers = n_lstm_layers
        self.lstm_dims = lstm_dims

        """model layers"""
        # input embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=n_embeddings,
            embedding_dim=embed_dims
        )

        # lstm layers
        self.lstm_layers = nn.LSTM(
            input_size=embed_dims,
            hidden_size=lstm_dims,
            num_layers=n_lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        # output projection layer
        self.output_layer = nn.Linear(lstm_dims * 2, n_classes)

    def forward(self, x, output_eval=False):
        # generate embeddings
        embeds = self.embedding_layer(x)  # (B, T) => (B, T, EMBED_DIM)

        # lstm forward pass
        lstm_op, _ = self.lstm_layers(embeds)  # (B, T, EMBED_DIM) => (B, T, 2*LSTM_DIMS)

        # project
        op_logits = self.output_layer(lstm_op)  # (B, T, 2*LSTM_DIMS) => (B, T, N_CLASSES)

        if not output_eval:
            return op_logits
        else:
            op = torch.argmax(F.softmax(op_logits, dim=-1), dim=-1)  # (B, T, N_CLASSES) => (B ,T)
            return op


if __name__ == "__main__":
    B, T = 32, 100
    model_config = {
        "n_classes": 10,
        "n_embeddings": 10000,
        "embed_dims": 69,
        "n_lstm_layers": 5,
        "lstm_dims": 12
    }

    inputs = torch.randint(
        low=0,
        high=model_config["n_embeddings"],
        size=(B, T)
    )
    expected_output_shape = (B, T, model_config["n_classes"])

    model = LSTMNER(**model_config)
    outputs = model(inputs, eval=False)

    assert outputs.shape == expected_output_shape
    print(f"input: {inputs.shape})")
    print(f"model config: ", model_config)
    print(f"output: {outputs.shape})")