from bs4 import BeautifulSoup

HTML = """
<html><body>
<ix:header>hidden metadata us-gaap:OtherAssets</ix:header>
<table><tr><td><a href="#risk">Item 1A.</a></td><td>Risk Factors</td></tr></table>
<div id="risk"><span style="font-weight:700">Item 1A.&#160;&#160;Risk Factors</span></div>
<p>Supply chains can fail. See Item 7 of this Form 10-K.</p>
<p>Apple Inc. | 2025 Form 10-K | 5</p>
<p>The Company&#8217;s demand can fall.</p>
</body></html>
"""

soup = BeautifulSoup(HTML, "html.parser")

print("--- 1. all text, nothing removed ---")
print(soup.get_text())

for tag in soup.find_all("ix:header"):
    tag.decompose()

print("--- 2.after removing the hidden header ---")
print(soup.get_text(separator="\n", strip=True))

print("--- 3. each paragraph ---")
for p in soup.find_all("p"):
    print(repr(p.get_text()))

print("--- 4. the link ---")
link = soup.find("a")
print(link["href"], "|", link.get_text())


print("--- 5. the heading ---")
heading = soup.find("span").get_text()
print(repr(heading))
print(repr(heading.replace("\xa0", " ")))


