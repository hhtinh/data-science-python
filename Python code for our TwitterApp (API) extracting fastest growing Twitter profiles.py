## Import Tweepy library for accessing the Twitter API: http://www.tweepy.org/
## For installing instructions: https://github.com/gmanual/Twitter-OAuth-GeekTool-Script/wiki/Installing-Tweepy-on-OSX
import tweepy

## Authentication details
## Step 1: Sign in with your Twitter Account : https://apps.twitter.com/
## Step 2: Create a new Twitter application (follow instructions)
## Step 3: Copy from API keys tab: consumer_key="API key" and consumer_secret="API secret"
## Step 4: Generate access token (API keys tab): access_token="Access token" and access_token_secret="Access token secret"

## Learn how to do: https://dev.twitter.com/oauth/overview/application-owner-access-tokens

## API keys look like:
consumer_key='aaaaBBBBBaaaaaaBBBBBBaaaaaBBBBBB'
consumer_secret='aaaBBBBBaaaaaBBBBBaaaaaaBBBBBBBaaaaaaBBBBBBBB'
access_token='ddddddddd-xxxXXxxXXXXXxxxxxXXXXxxxxxXXXXXxxxxxxxxx'
access_token_secret ='XXxxXXXXXxxxXXXxxXXXXxxxxxXXXXXxxxxxxxxx'

## Define user name
user='@datasciencectrl'

## Defining extract_twitter_data function which extracts data of twitter user
def extract_twitter_data(user):

    # Create authentication token
    # for more details visit the following webpage: http://tweepy.readthedocs.org/en/v2.3.0/auth_tutorial.html#auth-tutorial
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    ## Call the API and get data from twitter user
    api = tweepy.API(auth)
    twitterData = api.get_user(user)

    ## Print data
    print 'Twitter user: ' + user
    print 'Number of followers: ' + str(twitterData.followers_count)
    print 'Number of tweets: ' + str(twitterData.statuses_count)
    print 'Favorites: ' + str(twitterData.favourites_count)
    print 'Friends: ' + str(twitterData.friends_count)
    print 'Appears on ' + str(twitterData.listed_count) + ' lists'

## Call function: extract_twitter_data
extract_twitter_data(user)

###########################################################

## The output looks like:

Twitter user: @datasciencectrl

Number of followers: 10532

Number of tweets: 837

Favorites: 149

Friends: 1067

Appears on 376 lists