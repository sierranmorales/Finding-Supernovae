from astroquery.mast import Observations

#
#	Reads in Matchlog.txt and downloads images listed
#


f = open("Matchlog.txt", 'r')
lines = f.read().split('\n')
print(lines)

matchSet = set([])

for i in range(100):
	name1 = lines[i].split(" ")[0]
	name2 = lines[i].split(" ")[3]

	matchSet.add(name1)
	matchSet.add(name2)

f.close()

print(len(matchSet))

first_obs = Observations.query_criteria(obs_collection="HST", obs_id=matchSet)
data_product1 = Observations.get_product_list(first_obs)

manifest1 = Observations.download_products(data_product1, productType="SCIENCE", extension="flt.fits")