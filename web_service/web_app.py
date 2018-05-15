from web_service.comments_provider import CommentsProvider
from web_service.web_server import WebServer
from web_service.sa_consumer import SaConsumer


# Initialize and start the comments provider
comments_provider = CommentsProvider('AIzaSyB5XIRU9N6tj6q2Ea7bypaC96o0NNMXyW8', 'Movie Trailer', 10)
comments_provider.start()
print('Comments Provider started.')

# Initialize the WebApp and start the web server
web_server = WebServer()
web_server.start()
print('Web server started.')


def send_to_all_clients(msg):
    for client in web_server.get_all_connections():
        client.sendMessage(msg)

    print('A message has been broadcasted.')


# Initialize and start the kafka consumer (sentiment analysis)
sa_consumer = SaConsumer(send_to_all_clients)
sa_consumer.start()
print('SA Consumer started.')
