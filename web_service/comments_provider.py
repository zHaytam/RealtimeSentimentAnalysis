from youtube.youtube_scraper import YoutubeScraper
from kafka import KafkaProducer


class CommentsProvider:
    """
    Uses YoutubeScraper and provides comments using a KafkaProducer
    """

    def __init__(self, api_key, search_q, n_vids):
        self.youtube = YoutubeScraper(api_key, search_q, n_vids, self.__on_comment)
        self.producer = KafkaProducer(bootstrap_servers='localhost:9092')

    def __on_comment(self, video_id, comment):
        self.producer.send('test', bytes('{}||{}'.format(video_id, comment), 'utf-8'))
        print(video_id, comment)

    def start(self):
        self.youtube.fetch_videos()
        print('Video Ids:', self.youtube.videos_ids)
        self.youtube.start()
