import requests, json

def cookie_str_to_dict(cookie_str):

	return dict([line.strip().split('=', 1) for line in cookie_str.split(';')])


if __name__ == '__main__':
	url = 'http://compbio.charite.de/phenomizer/phenomizer/PhenomizerServiceURI'
	headers = {
		'Host': 'compbio.charite.de',
		'Connection': 'keep-alive',
		'Content-Length': '436',
		'Pragma': 'no-cache',
		'Cache-Control': 'no-cache',
		'X-GWT-Module-Base': 'http://compbio.charite.de/phenomizer/phenomizer/',
		'X-GWT-Permutation': '6B2D4F7E0EAFC0126A6CC15CB2E58BF8',
		'DNT': '1',
		'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
		'Content-Type': 'text/x-gwt-rpc; charset=UTF-8',
		'Accept': '*/*',
		'Origin': 'http://compbio.charite.de',
		'Referer': 'http://compbio.charite.de/phenomizer/',
		'Accept-Encoding': 'gzip, deflate',
		'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
		'Cookie': '_ga=GA1.2.795202033.1583922195; _gid=GA1.2.1607795462.1585028678',
	}

	data = '7|0|21|http://compbio.charite.de/phenomizer/phenomizer/|4D712CFAFF94336DB973772B66A9AB48|de.charite.phenontology.client.PhenomizerService|getSimilarOmimEntries|I|java.util.ArrayList/4159755760|java.lang.String/2004016611|Z|J|[Ljava.lang.String;/2600011424|HP:0005709|2-3 toe cutaneous syndactyly|observed.|HP:0004691|2-3 toe syndactyly|HP:0000878|11 pairs of ribs||pvalue|ASC|BH|1|2|3|4|9|5|5|6|7|8|7|7|7|9|60|30|6|3|10|3|11|12|13|10|3|14|15|13|10|3|16|17|13|18|0|19|20|21|XELLMzl|'
	r = requests.post(url, headers=headers, data=data)
	r.raise_for_status()
	print(r.text)

	pass


