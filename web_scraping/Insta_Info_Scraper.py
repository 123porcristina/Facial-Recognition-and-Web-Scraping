
#!/usr/bin/python
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
#scraper pulls data and shows the details on the screen
class Insta_Info_Scraper:

    def __init__(self, font, color, stroke, size):
        self.font = font
        self.color = color
        self.stroke = stroke
        self.size = size
        self.insta_dict = {}

    def check_info(self, user):
        return user in self.insta_dict

    def getinfo_dict(self, name ):

        if name not in self.insta_dict.values():
            return self.insta_dict[name]
        else:
            return "User not found"

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

        #####
        info_instagram = {name: {'User': user,
                                 'Followers': followers,
                                 'Following': following,
                                 'Posts': posts}}
        self.insta_dict.update(info_instagram)
        #####

        print('User:', user)
        print('Followers:', followers)
        print('Following:', following)
        print('Posts:', posts)
        print('---------------------------')
        # cv2.putText(frame, 'User:' + user, (x, y+h+15), font, size, color, stroke, cv2.LINE_AA)
        # cv2.putText(frame, 'Followers:'+ followers, (x, y+h+25), font, size, color, stroke, cv2.LINE_AA)
        # cv2.putText(frame, 'Following:'+ following, (x, y+h+35), font, size, color, stroke, cv2.LINE_AA)
        # cv2.putText(frame, 'Posts:'+ posts, (x, y+h+45), font, size, color, stroke, cv2.LINE_AA)
        # cv2.putText(frame, "Confidence" + str(round(conf)) + "%", (x, y+h+55), font, size, color, stroke, cv2.LINE_AA)
        # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


    def setTextScreen(self, frame, x, h, y, conf, w, name):

        dict_text = self.getinfo_dict(name)

        cv2.putText(frame, 'User:' + dict_text['User'], (x, y + h + 15), self.font, self.size, self.color, self.stroke, cv2.LINE_AA)
        cv2.putText(frame, 'Followers:' + dict_text['Followers'], (x, y + h + 25), self.font, self.size, self.color, self.stroke, cv2.LINE_AA)
        cv2.putText(frame, 'Following:' + dict_text['Following'], (x, y + h + 35), self.font, self.size, self.color, self.stroke, cv2.LINE_AA)
        cv2.putText(frame, 'Posts:' + dict_text['Posts'], (x, y + h + 45), self.font, self.size, self.color, self.stroke, cv2.LINE_AA)
        cv2.putText(frame, "Confidence" + str(round(conf)) + "%", (x, y + h + 55), self.font, self.size, self.color, self.stroke,
                    cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, 2)


    def main(self, frame, x,h,y,conf,w,name):
        val = self.check_info(name)
        if val == False:
            self.ctx = ssl.create_default_context()
            self.ctx.check_hostname = False
            self.ctx.verify_mode = ssl.CERT_NONE

            with open('users.txt') as f:
                self.content = f.readlines()
            self.content = [x.strip() for x in self.content]
            for url in self.content:
                self.getinfo(url, name)
                self.setTextScreen(frame, x,h,y,conf,w,name)
        else:
            self.setTextScreen(frame, x,h,y,conf,w,name)


if __name__ == '__main__':
    obj = Insta_Info_Scraper()
    obj.main()

# class Insta_Info_Scraper:
#
#     def getinfo(self, url):
#
#         html = urllib.request.urlopen(url, context=self.ctx).read()
#         print(url)
#         soup = BeautifulSoup(html, 'html.parser')
#         data = soup.find_all('meta', attrs={'property': 'og:description'
#                              })
#         text = data[0].get('content').split()
#         user = '%s %s' % (text[-3], text[-2])
#         followers = text[0]
#         following = text[2]
#         posts = text[4]
#         print ('User:', user)
#         print ('Followers:', followers)
#         print ('Following:', following)
#         print ('Posts:', posts)
#         print ('---------------------------')
#
#
#     def main(self):
#         self.ctx = ssl.create_default_context()
#         self.ctx.check_hostname = False
#         self.ctx.verify_mode = ssl.CERT_NONE
#
#         with open('users.txt') as f:
#             self.content = f.readlines()
#         self.content = [x.strip() for x in self.content]
#         for url in self.content:
#             self.getinfo(url)
#             print(url)
#
# if __name__ == '__main__':
#     obj = Insta_Info_Scraper()
#     obj.main()



