import flwr as fl
import csv
import os

# CSV file path
LOG_FILE = "training_logs.csv"

# Custom strategy to log metrics
class LoggingStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["round", "train_loss", "val_loss", "val_accuracy"])

    def aggregate_fit(self, rnd, results, failures):
        aggregated_result = super().aggregate_fit(rnd, results, failures)
        if aggregated_result is not None:
            loss = aggregated_result[1]["loss"] if "loss" in aggregated_result[1] else None
            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([rnd, loss, "", ""])
        return aggregated_result

    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        if aggregated_result is not None:
            loss, metrics = aggregated_result
            accuracy = metrics["accuracy"] if metrics and "accuracy" in metrics else None
            # Read previous train_loss entry for same round
            rows = []
            with open(LOG_FILE, mode='r') as f:
                rows = list(csv.reader(f))
            for i in range(len(rows)):
                if rows[i][0] == str(rnd):
                    rows[i][2] = f"{loss:.4f}"
                    rows[i][3] = f"{accuracy:.4f}" if accuracy is not None else ""
            # Rewrite the file
            with open(LOG_FILE, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        return aggregated_result

# Use custom strategy
def get_strategy():
    return LoggingStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=get_strategy()
    )
