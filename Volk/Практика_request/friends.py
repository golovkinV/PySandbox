import requests
from requests.exceptions import HTTPError
import constants as const
from datetime import datetime


def get_db_user_id(user_id):
    try:
        params = const.PAYLOAD
        params['user_ids'] = [user_id]
        req = requests.get(url=const.USERS, params=params)
        req.raise_for_status()
        data = req.json()
        print(data)
        return data['response'][0]['id']
    except HTTPError as http_error:
        print(f'HTTPError: {http_error}')
    except Exception as error:
        print(f'Exception: {error}')


def get_user_friends_ages(user_id):
    db_user_id = get_db_user_id(user_id=user_id)
    try:
        ages = []
        params = const.PAYLOAD
        params['user_id'] = db_user_id
        params['fields'] = const.FRIENDS_BDATE_FIELD
        req = requests.get(url=const.FRIENDS, params=params)
        req.raise_for_status()
        data = req.json()
        print(data)
        for friend in data['response']['items']:
            hasBDate = 'bdate' in friend
            if hasBDate:
                bdate = friend['bdate']
                bdate_params = bdate.split('.')
                hasBDateYear = len(bdate_params) == 3
                if hasBDateYear:
                    year = int(bdate_params[2])
                    age = datetime.now().year - year
                    ages.append(age)
        return ages
    except HTTPError as http_error:
        print(f'HTTPError: {http_error}')
    except Exception as error:
        print(f'Exception: {error}')


def calc_age(uid):
    ages = get_user_friends_ages(user_id=uid)
    age_count = {}
    for age in ages:
        if age in age_count:
            age_count[age] += 1
        else:
            age_count[age] = 1
    result = sorted(
        list(age_count.items()), key=lambda x: (-x[1], x[0])
    )
    return result


print(calc_age(uid=const.VK_ID))
