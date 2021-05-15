def delblankline(infile, outfile):
    infopen = open(infile, 'r', encoding="utf-8")
    outfopen = open(outfile, 'w', encoding="utf-8")
    db = infopen.read()
    outfopen.write(db.replace(' ', '\n'))
    infopen.close()
    outfopen.close()


delblankline("czh/answer.txt", "czh/answer.txt")