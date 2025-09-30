import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
 
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        scaled_logits = logits / temperature
        return scaled_logits

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # self.cuda()
        nll_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label,_,_ in valid_loader:
                if torch.isnan(input).any() or torch.isinf(input).any():
                    print("Skipping batch due to NaN or Inf values in input data or labels during calibration")
                    continue
                logits = self.model(input)
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    print("Skipping logits due to NaN or Inf values in logits during calibration")
                    continue
                logits_list.append(logits)
                labels_list.append(label)

            # SKIP padding for now i think its not needed
            # # Pad sequences to the same length
            # max_len = max([logit.size(1) for logit in logits_list])
            # logits_padded = [F.pad(logit, (0, 0, 0, max_len - logit.size(1))) for logit in logits_list]
            # # labels_padded = [F.pad(label, (0, max_len - label.size(1)), value=-100) for label in labels_list]
            # labels_padded = [F.pad(label, (0, max_len - label.size(-1)), value=-100) if label.dim() > 1 else F.pad(label, (0, max_len - label.size(0)), value=-100) for label in labels_list]
            logits = torch.cat(logits_list)
            labels = torch.cat(labels_list)

        # Flatten the logits and labels for the loss calculation
        logits_flat = logits.view(-1, logits.size(-1))  # [batch_size * num_segments, num_classes]
        labels_flat = labels.view(-1)  # [batch_size * num_segments]
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits_flat, labels_flat).item()
        before_temperature_ece = ece_criterion(logits_flat, labels_flat).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits_flat), labels_flat)
            print(f"Loss during optimization: {loss}")
            loss.backward()
            return loss
        optimizer.step(eval)
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits_flat), labels_flat).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits_flat), labels_flat).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece