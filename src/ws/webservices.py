import cherrypy
import cherrypy_cors
import json
import numpy as np
import re
from softmax import Softmax


@cherrypy.expose
class Classify(object):
	@cherrypy.tools.accept(media='text/plain')
	def GET(self):
		return "THIS IS GET"

	def POST(self, data):
		arr = re.split("\[|]|,", data)
		X = [[0 for x in range(28 * 28)] for y in range(1)]
		for i in range(28*28):
			X[0][i] = (255 - float(str(arr[i + 1]))) / 255.
		softmax = Softmax()
		y = softmax.classify(X)
		index = -1
		mv = -100
		strY = ""
		for i in range(10):
			if i > 0:
				strY = strY + ","
			strY = strY + str(y[0][i])
			if (y[0][i] * 1000) > mv:
				mv = (y[0][i] * 1000)
				index = i
		return strY
	'''
	def POST(self):
		return "THIS IS POST"
	'''

if __name__ == '__main__':
	cherrypy_cors.install()
	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
			'tools.sessions.on': True,
			'tools.response_headers.on': True,
			'cors.expose.on': True,
			'tools.response_headers.headers': [('Content-Type', 'text/plain')],
		}
	}
	cherrypy.quickstart(Classify(), '/', conf)
