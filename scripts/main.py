"""
Task 6 - Command-line system
============================
A transaction is shown only after both checks pass, and only if they name the
same person:

    face image -> face recognised -> product model runs -> voice clip
        -> voice matches that same person -> show the recommended product

Any failed check stops the transaction and the product is not shown.

Run from the repository root:
    python scripts/main.py                 # asks for the image, the voice, and a customer
    python scripts/main.py --demo          # one full transaction and three denied attempts
    python scripts/main.py --image images/raw/sheja_neutral.jpg --audio audio/raw/sheja_yesapprove.m4a --customer-id 151
"""

import argparse
import sys
from pathlib import Path

from face_model import UNKNOWN, face_details
from product_model import ROOT, default_customer_id, product_details
from voice_model import UNAUTHORIZED, voice_details

DEFAULT_IMAGE = ROOT / "images" / "raw" / "sheja_neutral.jpg"
DEFAULT_AUDIO = ROOT / "audio" / "raw" / "sheja_yesapprove.m4a"


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def run_transaction(image_path, audio_path, customer_id=None, transaction_id=None):
    """Face, then the product model, then voice. The product is returned only if both match."""
    face = face_details(image_path)
    result = {
        "image": str(image_path),
        "audio": str(audio_path),
        "face": face,
        "voice": None,
        "granted": False,
        "prepared": False,
        "product": None,
        "probabilities": None,
        "customer_id": None,
        "transaction_id": None,
        "reason": "",
    }
    if face["member"] == UNKNOWN:
        result["reason"] = "Face was not recognised. The product model was not called."
        return result
    if customer_id is None and transaction_id is None:
        raise ValueError("A customer id or a transaction id is required once the face is recognised.")

    details = product_details(customer_id=customer_id, transaction_id=transaction_id)
    result["prepared"] = True
    result["customer_id"] = details["customer_id"]
    result["transaction_id"] = details["transaction_id"]

    voice = voice_details(audio_path)
    result["voice"] = voice
    if voice["member"] == UNAUTHORIZED:
        result["reason"] = "Voice was not approved. The recommendation stays hidden."
        return result
    if voice["member"].casefold() != face["member"].casefold():
        result["reason"] = (
            f"Face is {face['member']} but the voice is {voice['member']}. "
            "Both checks must be the same person. The recommendation stays hidden."
        )
        return result

    result["granted"] = True
    result["product"] = details["product"]
    result["probabilities"] = details["probabilities"]
    result["reason"] = "Face and voice are the same person. The recommendation can be shown."
    return result


def print_result(result):
    face = result["face"]
    print(f"1. Face: {rel(result['image'])}")
    print(f"   {face['member']} ({face['reason']})")
    if not result["prepared"]:
        print(f"\nACCESS DENIED. {result['reason']}")
        return

    who = f"customer {result['customer_id']}, transaction {result['transaction_id']}"
    print(f"2. Product model ran for {who}. The recommendation is hidden until the voice matches.")

    voice = result["voice"]
    print(f"3. Voice: {rel(result['audio'])}")
    print(f"   {voice['member']} ({voice['reason']})")
    if not result["granted"]:
        print(f"\nACCESS DENIED. {result['reason']}")
        return

    print(f"\nACCESS GRANTED. {result['reason']}")
    print(f"Recommended product: {result['product']}")
    ranked = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
    for name, probability in ranked:
        print(f"   {name:<14} {probability:.2f}")


def ask(prompt, default):
    text = str(default)
    shown = rel(text) if Path(text).is_absolute() else text
    entered = input(f"{prompt} [{shown}]: ").strip()
    return entered or text


def interactive():
    print("Multimodal product recommendation")
    print("The recommendation is shown only if the face and the voice are the same enrolled member.\n")
    image = ask("Face image", DEFAULT_IMAGE)
    face = face_details(image)
    print(f"1. Face: {rel(image)}")
    print(f"   {face['member']} ({face['reason']})")
    if face["member"] == UNKNOWN:
        print("\nACCESS DENIED. Face was not recognised. The product model was not called.")
        return 1

    customer = ask("Customer id", default_customer_id())
    try:
        customer_id = int(customer)
    except ValueError:
        print(f"\nCustomer id must be an integer, got {customer!r}.")
        return 1
    audio = ask("Voice clip", DEFAULT_AUDIO)
    try:
        result = run_transaction(image, audio, customer_id=customer_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n{exc}")
        return 1
    # The face line was already printed while asking for the next input.
    print(f"2. Product model ran for customer {result['customer_id']}, "
          f"transaction {result['transaction_id']}. The recommendation is hidden until the voice matches.")
    voice = result["voice"]
    print(f"3. Voice: {rel(result['audio'])}")
    print(f"   {voice['member']} ({voice['reason']})")
    if result["granted"]:
        print(f"\nACCESS GRANTED. {result['reason']}")
        print(f"Recommended product: {result['product']}")
        ranked = sorted(result["probabilities"].items(), key=lambda item: item[1], reverse=True)
        for name, probability in ranked:
            print(f"   {name:<14} {probability:.2f}")
        return 0
    print(f"\nACCESS DENIED. {result['reason']}")
    return 1


def demo():
    """One granted transaction, plus a bad face, a bad voice, and a mismatched pair."""
    customer_id = default_customer_id()
    scenarios = [
        ("Unauthorized face", ROOT / "images" / "raw" / "unauthorized_face.jpg",
         ROOT / "audio" / "raw" / "sheja_yesapprove.m4a", False, False),
        ("Unauthorized voice", ROOT / "images" / "raw" / "sheja_neutral.jpg",
         ROOT / "audio" / "raw" / "unauthorized_yesapprove.m4a", True, False),
        ("Different person on the voice check", ROOT / "images" / "raw" / "sheja_neutral.jpg",
         ROOT / "audio" / "raw" / "aime_yesapprove.m4a", True, False),
        ("Full transaction", ROOT / "images" / "raw" / "sheja_neutral.jpg",
         ROOT / "audio" / "raw" / "sheja_yesapprove.m4a", True, True),
    ]
    failures = []
    for title, image, audio, expect_prepared, expect_granted in scenarios:
        print("=" * 72)
        print(title)
        print("=" * 72)
        try:
            result = run_transaction(image, audio, customer_id=customer_id)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            failures.append(title)
            continue
        print_result(result)
        prepared_ok = result["prepared"] is expect_prepared
        granted_ok = result["granted"] is expect_granted
        product_ok = (bool(result["product"]) if expect_granted else result["product"] is None)
        if prepared_ok and granted_ok and product_ok:
            print("\nScenario result: as expected.")
        else:
            print("\nScenario result: not what this demo expected.")
            failures.append(title)
        print()
    print("=" * 72)
    if failures:
        print("Demo failed: " + "; ".join(failures))
        return 1
    print("Demo passed: the full transaction was shown, and the three unauthorized attempts were denied.")
    return 0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Authenticate with a face and a matching voice, then show a product recommendation."
    )
    parser.add_argument("--demo", action="store_true",
                        help="Run a full transaction and the unauthorized attempts")
    parser.add_argument("--image", help="Face photo. With --audio, nothing is prompted.")
    parser.add_argument("--audio", help="Voice clip of an approved phrase.")
    parser.add_argument("--customer-id", type=int, help="Customer whose latest visit is scored.")
    parser.add_argument("--transaction-id", type=int, help="Score this visit instead of a customer's latest one.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.demo and any(value is not None for value in (args.image, args.audio, args.customer_id, args.transaction_id)):
        raise SystemExit("--demo uses its own samples. Leave out --image, --audio, --customer-id, and --transaction-id.")
    if args.demo:
        return demo()
    if args.image or args.audio or args.transaction_id is not None or args.customer_id is not None:
        if not args.image or not args.audio:
            raise SystemExit("Pass both --image and --audio, or run with no arguments to be prompted.")
        customer_id = args.customer_id if args.customer_id is not None else (
            None if args.transaction_id is not None else default_customer_id()
        )
        try:
            result = run_transaction(args.image, args.audio, customer_id=customer_id,
                                     transaction_id=args.transaction_id)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(str(exc))
        print_result(result)
        return 0 if result["granted"] else 1
    return interactive()


if __name__ == "__main__":
    sys.exit(main())
