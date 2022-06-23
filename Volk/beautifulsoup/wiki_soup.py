from bs4 import BeautifulSoup
import unittest
import re


class WikiParser:
    def __init__(self, content):
        self.content = content
        self.imgs = content.find_all("img")
        self.headers = content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        self.lists = content.find_all(['ul', 'ol'])

    def find_imgs_by_width(self, width: float):
        filter_lambda = lambda img: 'width' in img.attrs.keys() and float(img.attrs['width']) >= width
        return list(filter(filter_lambda, self.imgs))

    def find_headers_by_reg(self, reg):
        filter_lambda = lambda header: re.match(reg, header.text)
        return list(filter(filter_lambda, self.headers))

    def find_lists_without_parents(self):
        filter_lambda = lambda list: len(list.find_parents(name=['ul', 'ol'])) == 0
        return list(filter(filter_lambda, self.lists))

    def calc_a_sequence(self):
        links_len = 0
        content = self.content.find_next('a')
        while content:
            count = self.__calc_a_count(content.find_next_siblings())
            links_len = max(links_len, count)
            content = content.find_next('a')
        return links_len

    def __calc_a_count(self, sequence):
        count = 1
        for el in sequence:
            if el.name == 'a':
                count += 1
            else:
                break
        return count


def parse(path_to_file):
    f = open(path_to_file, encoding='utf-8')
    soup = BeautifulSoup(f.read(), features='html.parser')
    wiki_parser = WikiParser(content=soup.find("div", id="bodyContent"))
    imgs = wiki_parser.find_imgs_by_width(width=200)
    headers = wiki_parser.find_headers_by_reg(reg=r'^[ETC]+')
    links_len = wiki_parser.calc_a_sequence()
    lists = wiki_parser.find_lists_without_parents()
    return [len(imgs), len(headers), links_len, len(lists)]


class TestParse(unittest.TestCase):
    def test_parse(self):
        test_cases = (
            ('wiki/Stone_Age', [13, 10, 12, 40]),
            ('wiki/Brain', [19, 5, 25, 11]),
            ('wiki/Artificial_intelligence', [8, 19, 13, 198]),
            ('wiki/Python_(programming_language)', [2, 5, 17, 41]),
        )
        for path, expected in test_cases:
            with self.subTest(path=path, expected=expected): self.assertEqual(parse(path), expected)


if __name__ == '__main__':
    unittest.main()
