import requests


class YoutubeScraper:
    """
    Performs a Youtube Search, selects N videos (ordered by upload date) and monitors their comments.
    Previous comments will also be extracted.
    """

    SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'
    COMMENT_THREADS_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'

    def __init__(self, api_key, search_q, n_vids, regionCode=None):
        self.api_key = api_key
        self.search_q = search_q
        self.n_vids = 50 if n_vids > 50 else n_vids
        self.regionCode = regionCode
        self.videos_ids = None
        self.last_comment_per_video = None

    def __generate_search_params__(self):
        """
        Returns a parameters dictionary for the search query
        """
        params = {
            'key': self.api_key,
            'part': 'snippet',
            'maxResults': self.n_vids,
            'order': 'date',
            'type': 'video',
            'q': self.search_q
        }

        if self.regionCode is not None:
            params['regionCode'] = self.regionCode

        return params

    def fetch_videos(self):
        """
        Performs the Youtube Search and selects the top newest {n_vids} videos.
        """

        params = self.__generate_search_params__()
        json_result = requests.get(self.SEARCH_URL, params).json()

        if not json_result['items']:
            raise ValueError(json_result)

        self.videos_ids = [item['id']['videoId'] for item in json_result['items']]
        self.last_comment_per_video = {}

    def start_monitoring(self, callback, interval=5):
        """
        Starts the monitoring process with the given interval.
        The callback method is called everytime a new comment is retrieved
        """
        if self.videos_ids is None:
            raise ValueError('No video ids available, call fetch_videos first.')

        params = {
            'key': self.api_key,
            'part': 'snippet',
            'maxResults': 100,
            'order': 'time',
            'textFormat': 'plainText'
        }

        for video_id in self.videos_ids:
            params['videoId'] = video_id
            json_result = requests.get(self.COMMENT_THREADS_URL, params).json()

            if not json_result['items']:
                continue

            last_comment_id = None

            for item in json_result['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
                callback(video_id, comment)
                last_comment_id = item['id']

            self.last_comment_per_video[video_id] = last_comment_id
