from youtube.youtube_scraper import YoutubeScraper


ys = YoutubeScraper('AIzaSyB5XIRU9N6tj6q2Ea7bypaC96o0NNMXyW8', 'Movie Trailer', 5, regionCode='CA')

# search for the videos
ys.fetch_videos()
print(ys.videos_ids)

# monitor the comments
ys.start_monitoring(lambda video_id, comment: print(video_id, comment))