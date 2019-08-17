# !/usr/bin/python
# -*- coding: utf-8 -*-
import requests
import urllib.request
import urllib.parse
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json
import cv2


#######SCRAPER#################
# scraper pulls data and shows the details on the screen
class Insta_Info_Scraper:

    def __init__(self, font, color, stroke, size):
        self.font = font
        self.color = color
        self.stroke = stroke
        self.size = size
        self.insta_dict = {}

    # Verify if the info already exists in the dictionary
    def check_info(self, user):
        return user in self.insta_dict

    # Get info from the dictionary according to the user
    def getinfo_dict(self, name):
        # if the user exists returns the dictonary correspondent to it
        if name not in self.insta_dict.values():
            return self.insta_dict[name]
        else:
            return "User not found"

    # Get info by doing a request. In this case we are using Instagram.
    # This is called everytime is a new face on the screen
    def getinfo(self, url, name):

        html = urllib.request.urlopen(url, context=self.ctx).read()
        soup = BeautifulSoup(html, 'html.parser')
        data = soup.find_all('meta', attrs={'property': 'og:description'
                                            })
        text = data[0].get('content').split()
        user = '%s %s %s' % (text[-3], text[-2], text[-1])
        followers = text[0]
        following = text[2]
        posts = text[4]

        # add new info to the dictionary
        info_instagram = {name: {'User': user,
                                 'Followers': followers,
                                 'Following': following,
                                 'Posts': posts}}
        self.insta_dict.update(info_instagram)

        print('User:', user)
        print('Followers:', followers)
        print('Following:', following)
        print('Posts:', posts)
        print('---------------------------')

    # Set info about the user on the screen according to the face on it
    def setTextScreen(self, frame, x, h, y, conf, w, name):
        # retrieves info from the dictionary according to the user
        dict_text = self.getinfo_dict(name)

        cv2.putText(frame, 'User:' + dict_text['User'], (x, y-60), self.font, self.size, self.color, self.stroke,
                    cv2.LINE_AA)
        cv2.putText(frame, 'Followers:' + dict_text['Followers'], (x, y -45 ), self.font, self.size, self.color,
                    self.stroke, cv2.LINE_AA)
        cv2.putText(frame, 'Following:' + dict_text['Following'], (x, y -30), self.font, self.size, self.color,
                    self.stroke, cv2.LINE_AA)
        cv2.putText(frame, 'Posts:' + dict_text['Posts'], (x, y -15), self.font, self.size, self.color,
                    self.stroke, cv2.LINE_AA)
        cv2.putText(frame, "Confidence" + str(round(conf)) + "%", (x, y +h+15 ), self.font, self.size, self.color,
                    self.stroke,
                    cv2.LINE_AA)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, 2)

    def main(self, frame, x, h, y, conf, w, name):
        # it verifies if the info to get is new
        val = self.check_info(name)

        if val is False: # if the face is new get the info by doing a request
            self.ctx = ssl.create_default_context()
            self.ctx.check_hostname = False
            self.ctx.verify_mode = ssl.CERT_NONE

            with open('haar/users.txt') as f:
                self.content = f.readlines()
            self.content = [x.strip() for x in self.content]
            for url in self.content:
                self.getinfo(url, name)
                self.setTextScreen(frame, x, h, y, conf, w, name)
        else: # when the face is not new just get the info from the dictionary
            self.setTextScreen(frame, x, h, y, conf, w, name)


if __name__ == '__main__':
    obj = Insta_Info_Scraper()
    obj.main()
