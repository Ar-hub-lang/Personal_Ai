"""Read approved public HTTPS pages as bounded, inert text."""

import http.client
import ipaddress
import socket
import ssl
import time
from html.parser import HTMLParser
from urllib.parse import quote, urljoin, urlsplit

MAX_PAGE_BYTES = 1_000_000
MAX_DOCUMENT_CHARS = 100_000
REDIRECTS = {301, 302, 303, 307, 308}
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
BLOCKS = HEADINGS | {"p", "li", "div", "section", "article", "main", "br", "tr", "pre", "blockquote"}
HIDDEN = {"head", "script", "style", "template", "noscript", "nav", "footer", "svg"}


def validate_url(url: str):
    try:
        parts = urlsplit(url)
        if (parts.scheme != "https" or not parts.hostname or parts.username is not None
            or parts.password is not None or any(char.isspace() for char in url)
            or not url.isprintable() or "\\" in url or parts.port not in {None, 443}):
            raise ValueError
        parts.hostname.encode("idna")
    except ValueError as error:
        raise ValueError("Use a public HTTPS source URL without credentials or a custom port.") from error
    return parts


def public_addresses(hostname: str) -> list[str]:
    try:
        addresses = {
            ipaddress.ip_address(info[4][0])
            for info in socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        }
    except OSError as error:
        raise ValueError("The website hostname could not be resolved.") from error
    if not addresses or any(
        not address.is_global or address.is_multicast or address.is_reserved
        or (isinstance(address, ipaddress.IPv6Address) and (
            address.ipv4_mapped is not None or address.sixtofour is not None or address.teredo is not None
        ))
        for address in addresses
    ):
        raise ValueError("Website imports require a publicly routed HTTPS address.")
    return [str(address) for address in sorted(addresses, key=lambda address: address.version)]


def download_page(url: str) -> tuple[str | None, str, bytes]:
    parts = validate_url(url)
    host = parts.hostname.encode("idna").decode("ascii")
    addresses = public_addresses(host)
    target = quote(parts.path or "/", safe="/:@!$&'()*+,;=-._~%")
    if parts.query:
        target += "?" + quote(parts.query, safe="=&?/:;,+-._~%")
    context = ssl.create_default_context()
    # Connect to an address already checked above. The original host still controls TLS validation.
    last_error = None
    for address in addresses[:4]:
        connection = http.client.HTTPSConnection(host, timeout=8, context=context)
        try:
            with socket.create_connection((address, 443), timeout=5) as raw:
                tls = context.wrap_socket(raw, server_hostname=host)
                connection.sock = tls
                connection.request("GET", target, headers={
                    "Accept": "text/html, text/plain, text/markdown",
                    "Accept-Encoding": "identity",
                    "User-Agent": "Local-IT-Support/1.0",
                })
                with connection.getresponse() as response:
                    if response.status in REDIRECTS and (location := response.getheader("Location")):
                        return urljoin(url, location), "", b""
                    if response.status != 200:
                        raise ValueError(f"The website returned HTTP {response.status}.")
                    content_type = response.headers.get_content_type()
                    if content_type not in {"text/html", "text/plain", "text/markdown"}:
                        raise ValueError("Import an HTML, plain-text, or Markdown page.")
                    if response.getheader("Content-Encoding", "identity").casefold() != "identity":
                        raise ValueError("The website returned an unsupported encoding.")
                    data = bytearray()
                    deadline = time.monotonic() + 12
                    while len(data) <= MAX_PAGE_BYTES:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise ValueError("The website download timed out.")
                        tls.settimeout(min(8, remaining))
                        chunk = response.read1(min(65_536, MAX_PAGE_BYTES + 1 - len(data)))
                        if not chunk:
                            break
                        data.extend(chunk)
                    if len(data) > MAX_PAGE_BYTES:
                        raise ValueError("The website page exceeds the 1 MB input limit.")
                    charset = response.headers.get_content_charset() or "utf-8"
                    try:
                        text = data.decode(charset, errors="replace")
                    except LookupError as error:
                        raise ValueError("The website uses an unsupported text encoding.") from error
                    return None, content_type, text.encode()
        except (OSError, http.client.HTTPException) as error:
            last_error = error
        finally:
            connection.close()
    raise ValueError("Could not connect to the website over HTTPS.") from last_error


class ReadableHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden: list[str] = []
        self.lines: list[str] = []
        self.words: list[str] = []
        self.heading = False

    def flush(self):
        if text := " ".join("".join(self.words).split()):
            self.lines.append(("# " if self.heading else "") + text)
        self.words.clear()

    def handle_starttag(self, tag, attrs):
        if tag in HIDDEN:
            self.flush()
            self.hidden.append(tag)
        elif not self.hidden:
            if tag in BLOCKS:
                self.flush()
                self.heading = tag in HEADINGS
            elif tag in {"td", "th"}:
                self.words.append(" | ")

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        elif tag in BLOCKS:
            self.flush()
            self.heading = False

    def handle_data(self, data):
        if not self.hidden:
            self.words.append(data)


def read_website(url: str) -> tuple[str, str]:
    for _ in range(4):
        redirect, content_type, data = download_page(url)
        if redirect:
            url = redirect
            continue
        text = data.decode().strip()
        if content_type == "text/html":
            parser = ReadableHTML()
            parser.feed(text)
            parser.close()
            parser.flush()
            text = "\n".join(parser.lines)
        if not text or len(text) > MAX_DOCUMENT_CHARS:
            raise ValueError("The page must contain up to 100,000 characters of readable text.")
        return url, text
    raise ValueError("The website exceeded the redirect limit.")
