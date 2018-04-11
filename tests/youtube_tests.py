from youtube.youtube_scraper import YoutubeScraper


def test1():
    ys = YoutubeScraper('AIzaSyB5XIRU9N6tj6q2Ea7bypaC96o0NNMXyW8', 'Movie Trailer', 5,
                        lambda video_id, comment: print(video_id, comment), region_code='CA')

    # search for the videos
    ys.fetch_videos()
    print(ys.videos_ids)

    # monitor the comments
    ys.start()
    print(ys.last_comment_per_video)


def test2():
    ys = YoutubeScraper('AIzaSyB5XIRU9N6tj6q2Ea7bypaC96o0NNMXyW8', 'XXX', 1,
                        lambda video_id, comment: print(video_id, comment))
    ys.fetch_videos()

    # Set the video id manually to test
    ys.videos_ids = ['FdusB7xepE4']
    ys.last_comment_per_video['FdusB7xepE4'] = None

    ys.start()


test2()