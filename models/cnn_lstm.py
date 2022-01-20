import torch
import torch.nn as nn
import torch.nn.functional as F


class CnnLSTMNER(nn.Module):

    def __init__(
            self,
            n_classes,
            n_embeddings,
            embed_dims,
            n_cnn_layers,
            n_lstm_layers,
            n_cnn_channels=32,
            cnn_kernel_size=3,
            cnn_padding=1,
            lstm_dims=256
    ):
        super(CnnLSTMNER, self).__init__()

        """model hyperparameters"""
        self.n_classes = n_classes
        self.n_embeddings = n_embeddings
        self.embed_dims = embed_dims
        self.n_cnn_layers = n_cnn_layers
        self.n_lstm_layers = n_lstm_layers
        self.n_cnn_channels = n_cnn_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_padding = cnn_padding
        self.lstm_dims = lstm_dims

        """model layers"""
        # input embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=n_embeddings,
            embedding_dim=embed_dims
        )

        # cnn layers
        cnn_layers_list = [
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embed_dims if i == 0 else n_cnn_channels,
                    out_channels=n_cnn_channels,
                    kernel_size=cnn_kernel_size,
                    padding=cnn_padding
                ),
                nn.ReLU(),
                nn.BatchNorm1d(n_cnn_channels),
                nn.Dropout()
            )
            for i in range(n_cnn_layers)
        ]
        self.cnn_layers = nn.Sequential(*cnn_layers_list)

        # lstm layers
        self.lstm_layers = nn.LSTM(
            input_size=n_cnn_channels,
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
        embeds = embeds.transpose(2, 1)  # (B, T, EMBED_DIM) => (B, EMBED_DIM, T)

        # convolve
        cnn_op = self.cnn_layers(embeds)  # (B, EMBED_DIM, T) => (B, CNN_CHANNELS, T)
        cnn_op = cnn_op.transpose(2, 1)  # (B, CNN_CHANNELS, T) => (B, T, CNN_CHANNELS)

        # lstm forward pass
        lstm_op, _ = self.lstm_layers(cnn_op)  # (B, T, CNN_CHANNELS) => (B, T, 2*LSTM_DIMS)

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
        "n_cnn_layers": 5,
        "n_cnn_channels": 12,
        "n_lstm_layers": 5,
        "lstm_dims": 12
    }

    inputs = torch.randint(
        low=0,
        high=model_config["n_embeddings"],
        size=(B, T)
    )
    expected_output_shape = (B, T, model_config["n_classes"])

    model = CnnLSTMNER(**model_config)
    outputs = model(inputs, output_eval=False)

    assert outputs.shape == expected_output_shape
    print(f"input: {inputs.shape})")
    print(f"model config: ", model_config)
    print(f"output: {outputs.shape})")