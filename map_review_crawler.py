import argparse
import configparser
import logging
import re

import googlemaps
import pymongo
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

config = configparser.ConfigParser()
config.read('config.ini')
API_KEY = config['GOOGLEMAP']['KEY']

client = pymongo.MongoClient(host='mongodb://db:27017/')
db_map = client.google_map
places = db_map.places
reviews = db_map.reviews


def search_places(keyword, api_key, language, radius, region):
    gmaps = googlemaps.Client(key=API_KEY)
    return gmaps.places(keyword, language=language, region=region, radius=radius)

def parse_place(raw):
    result = {}
    result['place_id'] = raw.get('place_id')
    result['name'] = raw.get('name')
    result['address'] = raw.get('formatted_address')
    result['types'] = raw.get('types')
    result['price_level'] = raw.get('price_level')    
    return result

def get_nid(place_id):
    url = f'https://www.google.com/maps/place/?q=place_id:{place_id}'
    res = requests.get(url).text
    try:
        nid = re.findall('\[\\\\"\d+\\\\",\\\\"(\d+)\\\\"\]', res)[0]
    except IndexError:
        try:
            nid = re.findall('\[\\\\"\d+\\\\",\\\\"(-\d+)\\\\"\]', res)[0]
        except:
            logger.exception('Error occured while getting nid of place.')
            return
    except:
        return
    return nid

def get_places(keyword, language='zh-TW', region=None, radius=None):
    place_json = search_places(
        keyword=keyword,
        api_key=API_KEY,
        language=language,
        region=region,
        radius=radius)

    place_data = []
    for d in place_json['results']:
        parsed_data = parse_place(d)
        parsed_data['nid'] = get_nid(parsed_data['place_id'])
        if parsed_data['nid']:
            place_data.append(parsed_data)
            logger.info(f"Got Place Info. Name: {parsed_data['name']}, Place Id: {parsed_data['place_id']}")


    if place_data:
        places.insert_many(place_data)


# reviews
def get_reviews(nid, max=200, start=0, limit=100):
    place_name = db_map.places.find_one({'nid': nid})['name']
    
    base_url = 'https://www.google.com.tw'
    path = '/maps/preview/review/listentitiesreviews?authuser=0&hl=zh-TW&gl=tw&pb=!1m2!1y6795128465812342238!2y{nid}!2m2!1i{start}!2i{limit}!3e1!4m5!3b1!4b1!5b1!6b1!7b1!5m2!1skrCwYIiTH-zVmAXmprLQBg!7e81'

    count = 0
    reviews_data = []

    while count < max:
        if max - count < limit:
            limit = max - count
        url = base_url + path.format(nid=nid, start=start, limit=limit)
        res = requests.get(url)

        if res.ok:
            reviews_raw = split_reviews(res)
            if not reviews_raw:
                break  # crawled all reviews

            for r in reviews_raw:
                review = parse_review(r)
                if review:
                    review['nid'] = nid
                    reviews_data.append(review)
                else:
                    pass

            count += len(reviews_raw)
            start += limit

        else:
            logger.warning(f'Something wrong while crawling reviews on {place_name} (status code: {res.status_code}).')
            return

    if reviews_data:
        reviews.insert_many(reviews_data)
        logger.info(f"Crawled {count} reviews on {place_name}.")


def parse_review(review_raw):
    result = {}
    try:
        review_raw = eval(review_raw)
        result['user_name'] = review_raw[0][1]
        result['content'] = review_raw[3]
        result['rating'] = review_raw[4]
        result['user_id'] = review_raw[6]
        result['post_id'] = review_raw[10]
        result['published_time'] = review_raw[27]
    except:
        logger.exception('Error occured while parsing review.')
    return result

def find_last_end_index(start_indexes, res):
    left_brackets = 0
    right_brackets = 0
    start_index = start_indexes[-1]
    for i, s in enumerate(res[start_index:]):
        if s == '[':
            left_brackets += 1
        elif s == ']':
            right_brackets += 1
        else:
            pass
        if left_brackets - right_brackets == 0:
            break

    return (start_index + i + 1)

def get_review_indexes(res):
    start_indexes = []
    end_indexes = []

    for match in re.finditer('\[\["https://www.google.com/maps/contrib/', res):
        start_indexes.append(match.start())
        end_indexes.append(match.start() - 1)

    if end_indexes:
        end_indexes.pop(0)
        end_indexes.append(find_last_end_index(start_indexes, res))
    
    return start_indexes, end_indexes

def split_reviews(res):
    res = res.text.replace('null', '"null"').replace('\n', '')
    reviews = []
    start_indexes, end_indexes = get_review_indexes(res)
    if start_indexes:
        for start, end in zip(start_indexes, end_indexes):
            reviews.append(res[start: end])
    return reviews


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--place',
        default=None,
        help='The place to search on Google Map'
    )
    parser.add_argument(
        '-r', '--review',
        default=None,
        help='The place to search reviews on Google Map'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    
    if args.place:
        get_places(args.place)

    if args.review:
        for p in places.find({'name': {'$regex': args.review}}):
            nid = p['nid']
            get_reviews(nid, max=1000)