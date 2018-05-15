import requests
from threading import Thread, Event


class YoutubeScraper(Thread):
    """
    Performs a Youtube Search, selects N videos (ordered by upload date) and monitors their comments.
    Previous comments will also be extracted.
    """

    SEARCH_URL = 'https://www.googleapis.com/youtube/v3/search'
    COMMENT_THREADS_URL = 'https://www.googleapis.com/youtube/v3/commentThreads'

    def __init__(self, api_key, search_q, n_vids, callback, region_code=None, interval=5):
        self.stop_event = Event()
        Thread.__init__(self)
        self.api_key = api_key
        self.search_q = search_q
        self.n_vids = 50 if n_vids > 50 else n_vids
        self.callback = callback
        self.regionCode = region_code
        self.interval = interval
        self.videos_ids = None
        self.last_comment_per_video = None

    def __generate_search_params(self):
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

    def __generate_comment_threads_params(self, page_token=None):
        """
        Returns a parameters dictionary for the comment threads query
        """
        params = {
            'key': self.api_key,
            'part': 'snippet',
            'maxResults': 100,
            'order': 'time',
            'textFormat': 'plainText'
        }

        if page_token is not None:
            params['pageToken'] = page_token

        return params

    def fetch_videos(self):
        """
        Performs the Youtube Search and selects the top newest {n_vids} videos.
        """
        params = self.__generate_search_params()
        json_result = requests.get(self.SEARCH_URL, params).json()

        if not json_result['items']:
            raise ValueError(json_result)

        self.videos_ids = []
        self.last_comment_per_video = {}

        for item in json_result['items']:
            video_id = item['id']['videoId']
            self.videos_ids.append(video_id)
            self.last_comment_per_video[video_id] = []

    def __extract_comments(self, video_id, page_token=None):
        """
        Performs the comment threads request and calls callback for each comment.
        Returns the json_result.
        """
        params = self.__generate_comment_threads_params(page_token)
        params['videoId'] = video_id
        json_result = requests.get(self.COMMENT_THREADS_URL, params).json()

        if 'items' not in json_result or len(json_result['items']) == 0:
            return None

        for item in json_result['items']:
            comment_id = item['id']

            # In case we reached the last comment registred
            if len(self.last_comment_per_video[video_id]) > 0 and \
                    comment_id == self.last_comment_per_video[video_id][0]:
                break

            # Ignore the comments we already have (in case someone deletes his comment)
            if comment_id in self.last_comment_per_video[video_id]:
                continue

            self.last_comment_per_video[video_id].append(comment_id)
            comment = item['snippet']['topLevelComment']['snippet']['textOriginal']
            self.callback(video_id, comment)

        return json_result

    def __check_for_new_comments(self):
        """
        Checks if there is new comments in the videos
        """
        for video_id in self.videos_ids:
            json_result = self.__extract_comments(video_id)

    def run(self):
        """
        Starts the monitoring process with the given interval.
        The callback method is called everytime a new comment is retrieved
        """
        if self.videos_ids is None:
            raise ValueError('No video ids available, call fetch_videos first.')

        for video_id in self.videos_ids:
            json_result = self.__extract_comments(video_id)

            if json_result is None:
                self.last_comment_per_video[video_id] = []
                print('{} has no comments.'.format(video_id))
                continue

            # Check if there are next pages
            while 'nextPageToken' in json_result:
                json_result = self.__extract_comments(video_id, json_result['nextPageToken'])

        # Start monitoring
        print('Started monitoring')
        while not self.stop_event.wait(self.interval):
            self.__check_for_new_comments()

    def stop(self):
        """
        Sets the stop_event
        """
        self.stop_event.set()
