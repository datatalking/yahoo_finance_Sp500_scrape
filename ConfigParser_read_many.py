# TODO https://www.jetbrains.com/help/pycharm/run-debug-configuration-python.html#1
# SOURCE Python Standard Library - Doug Hellman
# TODO setup sqlserver config stuff https://www.linode.com/docs/guides/securely-manage-remote-postgresql-servers-with-pgadmin-on-macos-x/

from ConfigParser import SafeConfigParser
import codecs

parser = SafeConfigParser()
parser.read('multisection.ini')

for section_name in parser.sections():
	print('Section:', section_name)
	print(' Options:', parser.options(section_name))
	for name, value in parser.items(section_name):
		print(' %s = %s' % (name, value))
	print()


# open file with correct encoding
with codecs.open('unicode.ini', 'r', encoding='utf-8') as f:
	parser.readfp(f)

password = parser.get('bug_tracker', 'password')

print('Password:', password.encode('utf-8'))
print()