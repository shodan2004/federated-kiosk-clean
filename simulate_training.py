import time
import random
import psycopg2

# PostgreSQL connection
conn = psycopg2.connect(
    dbname="federated_logs",
    user="admin",
    password="admin",
    host="localhost",
    port="5433"
)
cur = conn.cursor()

clients = ["Kiosk_01", "Kiosk_02", "Kiosk_03"]
rounds = 1

while True:
    for client in clients:
        train_loss = round(random.uniform(0.3, 1.0), 4)
        val_loss = round(train_loss - random.uniform(0.01, 0.1), 4)
        train_accuracy = round(random.uniform(0.65, 0.9), 4)
        val_accuracy = round(train_accuracy + random.uniform(0.01, 0.07), 4)
        learning_rate = round(random.uniform(0.0001, 0.01), 5)

        cur.execute("""
            INSERT INTO training_logs (client_id, round, train_loss, val_loss, val_accuracy, train_accuracy, learning_rate)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (client, rounds, train_loss, val_loss, val_accuracy, train_accuracy, learning_rate))

        conn.commit()
        print(f"Inserted round {rounds} for {client}")
        time.sleep(1)

    rounds += 1
    time.sleep(3)
