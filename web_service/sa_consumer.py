from threading import Thread
from kafka import KafkaConsumer


class SaConsumer:
    """
    Creates a Kafka Consumer to consume Sentiment Analysis and sends it to the web server's clients
    """

    def __init__(self, on_message):
        self._consumer = KafkaConsumer('sa')
        self._thread = Thread(target=self.__poll_messages)
        self.on_message = on_message

    def __poll_messages(self):
        for msg in self._consumer:
            self.on_message(msg.value)

    def start(self):
        self._thread.start()
