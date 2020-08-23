import torch

from slr_model import Encoder, Decoder, Seq2Seq


def main():
    device = torch.device("cuda")
    encoder = Encoder(lstm_hidden_size=512)
    decoder = Decoder(output_dim=500, emb_dim=256, enc_hid_dim=512, dec_hid_dim=512, dropout=0.5)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).cuda()
    model.eval()
    imgs = torch.randn(16, 3, 8, 128, 128).cuda()
    target = torch.LongTensor(16, 8).random_(0, 500).cuda()
    dummy_input = imgs, target
    torch.onnx.export(model, dummy_input, 'slr.onnx', verbose=True)

if __name__ == '__main__':
    main()