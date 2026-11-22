"""Hand-written building blocks for the synthetic ticket generator (used by data_prep.py).

How a ticket is assembled
-------------------------
1. Pick a category, then a priority from that category's priority distribution.
2. Pick an :class:`Issue` of that category that is plausible at that priority.
3. Priority is expressed through three latent *factors*, each rendered with varied wording
   (or left implicit):
     - scope:    who is affected (just me / my team / whole office / customers / a VIP)
     - blocking: workaround exists / slowed down / completely blocked
     - deadline: none / this week / today / right now
4. Wrap it in a writing style (formal, casual, terse, frustrated, ...) and add typos.

Rule for editing: within one Issue, every subject must fit every problem (they are drawn
independently). Problem texts deliberately mention other categories' vocabulary ("Outlook"
in a Wi-Fi ticket, "password" in a phishing ticket) so a classifier must read for the root cause.

Placeholders like ``{app}`` are filled from :data:`SLOTS`; one value per slot per ticket.
If you change the label set in config.yaml, add/rename the matching keys here
(or generate data for new labels with ``data_prep.py --llm``).
"""

from __future__ import annotations

from dataclasses import dataclass

ALL_PRIORITIES = ("Low", "Medium", "High", "Critical")
LMH = ("Low", "Medium", "High")
LM = ("Low", "Medium")


@dataclass(frozen=True)
class Issue:
    """One coherent kind of problem within a category."""

    subjects: tuple[str, ...]
    problems: tuple[str, ...]
    details: tuple[str, ...] = ()
    priorities: tuple[str, ...] = ALL_PRIORITIES
    # "single":   one person/device (Critical only via a VIP / customer-facing person).
    # "shared":   a service many people use (Critical via office-wide or customer impact).
    # "incident": a security incident (Critical when it spreads or touches sensitive data).
    scale: str = "single"


# --------------------------------------------------------------------------- label balance
# Realistic imbalance: lots of software/access tickets, few security/other; Critical is rare.
CATEGORY_WEIGHTS: dict[str, float] = {
    "Software": 0.19, "Access/Accounts": 0.17, "Hardware": 0.14, "Network": 0.13,
    "Email": 0.12, "Printer": 0.09, "Security": 0.08, "Other": 0.08,
}
PRIORITY_GIVEN_CATEGORY: dict[str, dict[str, float]] = {
    "Hardware": {"Low": 0.25, "Medium": 0.40, "High": 0.28, "Critical": 0.07},
    "Software": {"Low": 0.22, "Medium": 0.43, "High": 0.27, "Critical": 0.08},
    "Network": {"Low": 0.12, "Medium": 0.35, "High": 0.33, "Critical": 0.20},
    "Access/Accounts": {"Low": 0.20, "Medium": 0.40, "High": 0.32, "Critical": 0.08},
    "Email": {"Low": 0.25, "Medium": 0.40, "High": 0.25, "Critical": 0.10},
    "Security": {"Low": 0.10, "Medium": 0.25, "High": 0.35, "Critical": 0.30},
    "Printer": {"Low": 0.35, "Medium": 0.42, "High": 0.19, "Critical": 0.04},
    "Other": {"Low": 0.55, "Medium": 0.37, "High": 0.08, "Critical": 0.0},
}

# --------------------------------------------------------------------------- slot values
SLOTS: dict[str, tuple[str, ...]] = {
    "device": ("Dell Latitude 7440", "ThinkPad T14", "MacBook Pro", "HP EliteBook 840", "Surface Pro 9",
               "Dell XPS 13", "ThinkPad X1 Carbon", "work laptop", "desktop PC"),
    "monitor": ("Dell 27 inch monitor", "second monitor", "external display", "LG ultrawide", "left monitor"),
    "peripheral": ("wireless mouse", "keyboard", "headset", "webcam", "Jabra headset", "Logitech webcam",
                   "laptop charger", "USB-C hub"),
    "app": ("Excel", "Word", "PowerPoint", "Teams", "Zoom", "Adobe Acrobat", "AutoCAD", "Power BI",
            "Visual Studio Code", "Tableau", "QuickBooks", "Slack"),
    "app_install": ("Visio", "Python", "Tableau Desktop", "Adobe Photoshop", "Power BI Desktop", "Notepad++",
                    "7-Zip", "Postman", "MATLAB", "Zoom", "Docker Desktop"),
    "system": ("SAP", "Salesforce", "Workday", "the CRM", "the ERP", "Jira", "Confluence", "the expenses portal",
               "the HR portal", "NetSuite", "the booking system"),
    "browser": ("Chrome", "Edge", "Firefox", "Safari"),
    "printer": ("the HP LaserJet on the 3rd floor", "the big Xerox by reception", "the Canon copier in the mailroom",
                "the printer next to the kitchen", "the colour printer in Marketing", "the Ricoh on floor 2"),
    "floor": ("2nd floor", "3rd floor", "4th floor", "ground floor", "5th floor"),
    "site": ("London", "Manchester", "Pune", "Austin", "Toronto", "Berlin", "Dublin", "Singapore", "Leeds"),
    "team": ("Finance", "HR", "Sales", "Marketing", "Legal", "Customer Support", "Engineering", "Operations",
             "Procurement", "Payroll", "Design", "Logistics"),
    "time": ("10am", "11:30", "2pm", "3pm", "4:30", "noon", "9am tomorrow"),
    "n": ("3", "4", "5", "6", "8", "10", "12", "15"),
    "big_n": ("80", "120", "200", "350", "500"),
    "since": ("since this morning", "since yesterday", "since the update last night", "for the past two days",
              "since I came back from leave", "since Monday", "on and off all week", "for about an hour"),
    "person": ("Priya", "Tom", "Aisha", "Daniel", "Mei", "Carlos", "Fatima", "James", "Olga", "Rahul", "Sophie",
               "Kwame", "Lucas", "Hannah", "Arjun", "Elena", "Omar", "Grace", "Yuki", "Ben"),
    "shared_drive": ("the Finance shared drive", "the S: drive", "the Projects folder on SharePoint",
                     "the HR SharePoint site", "the Contracts folder", "the Marketing team files"),
    "dl": ("the sales-team distribution list", "the all-staff list", "the finance-approvals group",
           "the support@ shared mailbox", "the invoices@ mailbox"),
    "vpn": ("VPN", "GlobalProtect", "Cisco AnyConnect", "FortiClient", "the company VPN"),
    "os_update": ("Windows update", "macOS update", "latest patch", "update IT pushed last night"),
    "room": ("3A", "Orion", "the boardroom", "4.12", "Thames", "Everest", "the training room"),
    "customer_action": ("place orders", "log in to the portal", "check out on the website", "reach our support line",
                        "download their invoices", "book appointments"),
    "day": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "tomorrow"),
    "phone": ("ext. 4417", "07700 900123", "ext. 2290", "my mobile"),
}

# --------------------------------------------------------------------------- category issues
CATEGORY_ISSUES: dict[str, tuple[Issue, ...]] = {
    "Hardware": (
        Issue(
            subjects=("Laptop won't turn on", "{device} not powering on", "laptop dead", "Can't start my laptop"),
            problems=("My {device} won't turn on at all, the power light doesn't come on.",
                      "I press the power button and nothing happens, even when it's plugged in.",
                      "The laptop shows a black screen after the logo and then switches off.",
                      "My {device} won't start, it just beeps three times when I press the power button."),
            details=("I've tried a different charger.", "I held the power button for 30 seconds like the guide says.",
                     "The charging light blinks orange.", "It was fine when I shut it down yesterday."),
            priorities=("Medium", "High", "Critical"),
        ),
        Issue(
            subjects=("Screen problem", "Display problem on {device}", "laptop screen issue",
                      "Something wrong with my screen"),
            problems=("My laptop screen flickers, especially when I join Teams calls.",
                      "I dropped my {device} and now the screen has a crack across the corner.",
                      "There are coloured lines across my screen and it's getting worse.",
                      "The screen goes dim and I can't change the brightness."),
            details=("It's fine when I connect an external monitor.", "Restarting doesn't help.",
                     "It started {since}."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Docking station problem", "External monitor issue", "dock problem", "Screens not working with dock"),
            problems=("My {monitor} isn't detected when I connect through the docking station.",
                      "The dock stopped charging my laptop and the screens stay black.",
                      "My external display goes black every few minutes when I'm docked.",
                      "Only one of my two monitors works through the dock {since}."),
            details=("I've swapped the HDMI cable already.", "The dock's light is on but nothing happens.",
                     "It works if I plug the monitor straight into the laptop."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Need a new {peripheral}", "{peripheral} broken", "Request: replacement {peripheral}",
                      "{peripheral} not working"),
            problems=("My {peripheral} stopped working, I think it's broken.",
                      "Could I get a new {peripheral}? Mine is falling apart.",
                      "My {peripheral} only works some of the time, I have to keep unplugging it.",
                      "My laptop doesn't recognise my {peripheral} anymore."),
            details=("I tried it on another USB port.", "A colleague's one works on my laptop, so it's the device.",
                     "It's about four years old."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Laptop battery draining fast", "Battery problem", "Battery not holding charge"),
            problems=("My battery goes from 100% to 20% in about an hour.",
                      "The battery icon says 'plugged in, not charging'.",
                      "The trackpad is lifting up, I think the battery might be swollen."),
            details=("Battery health in settings says 'Service recommended'.", "The laptop is about three years old."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Laptop overheating", "Fan very loud", "Laptop getting very hot"),
            problems=("The laptop gets really hot and the fan is loud all the time.",
                      "My {device} gets so hot it switches itself off during video calls."),
            details=("It happens even with only the browser open.", "I keep it on a desk, not on my lap."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Laptop for new starter", "Hardware for new hire", "Equipment request for {person}"),
            problems=("{person} starts in {team} next week and will need a laptop and two monitors.",
                      "We have a new hire starting on {day}, can we get a {device} set up for them?",
                      "Please order a docking station and headset for our new team member {person}."),
            details=("Their manager has approved it.", "They'll be based in the {site} office.",
                     "Same setup as the rest of {team} please."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Phone problem", "Issue with my work phone", "work phone not working"),
            problems=("My desk phone shows 'No service' and I can't make calls.",
                      "The company iPhone I was given won't charge anymore.",
                      "My work mobile's screen cracked and the touch doesn't respond in one corner."),
            details=("I tried a different cable and plug.", "It's the phone I use for the on-call rota."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Problem in meeting room {room}", "Meeting room equipment not working", "{room} AV issue"),
            problems=("The big screen in meeting room {room} shows nothing when we plug in.",
                      "The conference camera in {room} isn't working.",
                      "The projector in {room} keeps turning itself off after a few minutes."),
            details=("We tried both HDMI cables on the table.", "The remote has new batteries.",
                     "The room panel says the device is offline."),
            priorities=LMH,
            scale="shared",
        ),
        Issue(
            subjects=("Hard drive problem?", "{device} won't boot properly", "Laptop disk issue"),
            problems=("My laptop is making a clicking noise and files take forever to open.",
                      "Windows says 'No bootable device found' when I start up.",
                      "My laptop says the disk is failing and to back up my files."),
            details=("I haven't backed up my desktop folder recently.", "I turned it off straight away."),
            priorities=("Medium", "High", "Critical"),
        ),
        Issue(
            subjects=("Spilled coffee on laptop", "Liquid spill on keyboard", "Water on my laptop"),
            problems=("I spilled coffee on my keyboard and now some keys don't work.",
                      "I knocked a glass of water over my {device} and it switched off."),
            details=("I turned it off and turned it upside down.", "It smells a bit burnt."),
            priorities=("Medium", "High", "Critical"),
        ),
    ),
    "Software": (
        Issue(
            subjects=("{app} keeps crashing", "{app} crash", "{app} not responding", "problem with {app}"),
            problems=("{app} crashes every time I try to open a file.",
                      "{app} freezes and then closes without any error message.",
                      "Whenever I open {app} it says 'Not responding' and I have to force quit it.",
                      "{app} crashes when I open files from {shared_drive}."),
            details=("I've restarted my laptop twice.", "I tried repairing Office from the Control Panel.",
                     "It started after the {os_update}.",
                     "The error says 'The application was unable to start correctly (0xc000007b)'."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Software install request: {app_install}", "Please install {app_install}",
                      "Need {app_install} installed", "{app_install} request"),
            problems=("Could you please install {app_install} on my laptop? I need it for a project.",
                      "I need {app_install} but the installer asks for an admin password.",
                      "Requesting {app_install} for my role, I don't have it in Software Center."),
            details=("My manager has approved this.", "It's in the approved software list I think.",
                     "I'm on a Mac if that matters."),
            priorities=LMH,
        ),
        Issue(
            subjects=("{app} licence expired", "Activation error in {app}", "{app} says unlicensed"),
            problems=("{app} says my licence has expired and it's now in read-only mode.",
                      "{app} keeps asking me to sign in and then says I don't have a licence.",
                      "{app} shows 'Unlicensed Product' across the top."),
            details=("I signed out and back in.", "It worked fine last week."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Update problem", "Update stuck", "Problem after update", "update failed"),
            problems=("My laptop has been stuck on 'Working on updates 35%' for an hour.",
                      "The {os_update} failed with error 0x800f0922 and keeps retrying.",
                      "After the {os_update} my laptop is really slow and apps take ages to open."),
            details=("I left it overnight and it was still stuck.", "There's plenty of disk space."),
            priorities=LMH,
        ),
        Issue(
            subjects=("{system} error", "{system} not loading", "Problem with {system}", "{system} down?"),
            problems=("{system} shows 'Internal Server Error' when I try to submit a form.",
                      "{system} is extremely slow today, pages take over a minute to load while other sites are fine.",
                      "I get a blank white page when I open {system}.",
                      "Reports in {system} are still showing last week's data."),
            details=("I cleared my browser cache.", "It does the same thing in a different browser.",
                     "Screenshot attached."),
            priorities=ALL_PRIORITIES,
            scale="shared",
        ),
        Issue(
            subjects=("{browser} problems", "Browser issue", "Problem with {browser}"),
            problems=("{browser} keeps crashing when I have more than a few tabs open.",
                      "{browser} says my profile could not be loaded.",
                      "Some buttons in {system} don't work in {browser} but do work in another browser."),
            details=("I've disabled my extensions.", "I reinstalled it yesterday."),
            priorities=LM,
        ),
        Issue(
            subjects=("Excel problem", "Issue with an Excel file", "Excel error"),
            problems=("Our month-end Excel macro stopped working after the update, it gives a 'Compile error'.",
                      "Excel says the file is corrupt and can't be opened.",
                      "Excel formulas show #REF! after I copied the sheet to a new workbook."),
            details=("Other files open fine.", "The file is on {shared_drive}.", "It's an .xlsm file."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Word formatting issue", "Word keeps changing my document", "Problem with Word"),
            problems=("Word keeps changing my document formatting every time I save.",
                      "Track changes in Word won't turn off on a contract I'm editing."),
            details=("It only happens with this one document.",),
            priorities=LMH,
        ),
        Issue(
            subjects=("Teams won't start", "Teams stuck loading", "Problem with the Teams app"),
            problems=("Teams is stuck on the loading screen with the logo spinning.",
                      "Teams keeps signing me out and telling me to update, but the update fails."),
            details=("I cleared the Teams cache like the intranet article says.", "Reinstalling didn't help."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Zoom can't find microphone", "No audio in Zoom", "Zoom audio problem"),
            problems=("Zoom can't find my microphone even though it works in other apps.",
                      "People can't hear me in Zoom, but the mic test in Windows works."),
            details=("I picked the right device in Zoom settings.",),
            priorities=LMH,
        ),
        Issue(
            subjects=("Computer running slow", "Laptop very slow", "Something using all the CPU"),
            problems=("My computer is very slow, it takes 10 minutes to start and every program lags.",
                      "Something is using 100% CPU and my laptop is basically unusable.",
                      "Task Manager shows a process called 'Antimalware Service' using most of the memory."),
            details=("I only have Outlook and Excel open.", "I've restarted it a few times."),
            priorities=LMH,
        ),
    ),
    "Network": (
        Issue(
            subjects=("Wi-Fi problem", "wifi issues", "Wi-Fi not working properly", "bad wifi at my desk"),
            problems=("My Wi-Fi keeps disconnecting every few minutes.",
                      "I can see the corporate Wi-Fi but it won't connect, it just says 'Can't connect to this network'.",
                      "Wi-Fi is really slow at my desk on the {floor}.",
                      "Teams calls keep dropping and websites time out, I think it's the Wi-Fi."),
            details=("Other devices seem fine.", "I forgot the network and re-added it.",
                     "It's fine when I use my phone hotspot."),
            priorities=LMH,
        ),
        Issue(
            subjects=("{vpn} not working", "VPN problem", "Can't use VPN from home"),
            problems=("{vpn} fails with 'Connection attempt has failed' when I work from home.",
                      "The VPN connects but then I can't reach any internal sites.",
                      "{vpn} disconnects every 10 to 15 minutes.",
                      "I get error 809 when I try to connect to the VPN."),
            details=("My home internet works fine otherwise.", "I restarted my router.",
                     "It worked yesterday from the same place."),
            priorities=ALL_PRIORITIES,
        ),
        Issue(
            subjects=("Network down", "No network", "Network outage", "Internet down in the office"),
            problems=("We can't get online on the {floor}, wired or wireless.",
                      "The internet is down in the {site} office.",
                      "All the network ports in our area stopped working after the power cut.",
                      "Internal sites and {system} are unreachable from the {site} office, but the internet works."),
            details=("The switch in the comms cupboard has orange lights.", "Phones on the network are also down."),
            priorities=("High", "Critical"),
            scale="shared",
        ),
        Issue(
            subjects=("Network very slow", "Slow network", "Slow internet today"),
            problems=("Everything online is really slow today, copying files to {shared_drive} takes ages.",
                      "Downloads crawl at about 1 Mbps in the office.",
                      "Video calls are choppy and freeze for people in the {team} area."),
            details=("Speed test attached.", "It's worse in the afternoon."),
            priorities=LMH,
            scale="shared",
        ),
        Issue(
            subjects=("Network port not working", "No connection at my desk", "Ethernet issue"),
            problems=("The network port at my new desk doesn't work, the light by the cable never comes on.",
                      "When I plug in the ethernet cable it says 'Unidentified network'."),
            details=("I tried a different cable.", "The desk number is on the label: {floor}-17."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Can't reach website", "DNS error", "Intranet not loading"),
            problems=("I get ERR_NAME_NOT_RESOLVED for the intranet but Google works.",
                      "Some websites won't load at all, others are fine.",
                      "The intranet works if I type the IP address but not the name."),
            details=("Same in {browser} and another browser.", "My colleague's laptop does it too."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Guest Wi-Fi", "Guest wifi for visitors", "Visitor Wi-Fi access"),
            problems=("We have visitors coming on {day} who will need guest Wi-Fi.",
                      "The guest Wi-Fi portal isn't sending the access code to our visitors."),
            details=("They're from a client, around 6 people.",),
            priorities=LMH,
        ),
        Issue(
            subjects=("{site} site offline", "Site network down", "Remote site has no network"),
            problems=("The {site} warehouse lost its network connection and the handheld scanners can't sync.",
                      "The card machines in our {site} shop can't connect because the network is down.",
                      "The link to the {site} office seems to be down, they can't reach any of our systems."),
            details=("The router lights are all red.", "The internet provider says there's no fault on their side."),
            priorities=("High", "Critical"),
            scale="shared",
        ),
    ),
    "Access/Accounts": (
        Issue(
            subjects=("Password problem", "Can't log in", "Password reset needed", "login issue"),
            problems=("I forgot my password and I'm locked out of my laptop.",
                      "My password expired while I was on leave and now I can't log in.",
                      "It says my password is incorrect but I'm sure it's right.",
                      "Outlook and Teams keep asking for my password and won't accept it, I think my account is locked."),
            details=("I tried the self-service reset but it says my phone number isn't registered.",
                     "Caps lock is off.", "I changed it on Friday."),
            priorities=("Medium", "High", "Critical"),
        ),
        Issue(
            subjects=("Account locked out", "Locked out again", "Account keeps locking"),
            problems=("My account is locked after too many attempts.",
                      "I keep getting locked out every morning even after resetting my password."),
            details=("I think my old phone might still have my email set up with the old password.",),
            priorities=("Medium", "High", "Critical"),
        ),
        Issue(
            subjects=("MFA problem", "MFA not working", "Authenticator issue"),
            problems=("I got a new phone and the Authenticator app no longer has my work account.",
                      "The MFA prompt never arrives on my phone.",
                      "It says my verification code is wrong every time."),
            details=("My old phone was wiped.", "The time on my phone is set automatically."),
            priorities=("Medium", "High", "Critical"),
        ),
        Issue(
            subjects=("Access to {shared_drive}", "Folder permissions", "Shared folder access"),
            problems=("I need access to {shared_drive} for the audit.",
                      "I get 'Access denied' when I open {shared_drive}, I had access last week.",
                      "Can I get edit rights on {shared_drive}? I only have read-only."),
            details=("My manager {person} has approved it.", "I moved from {team} last month."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Access to {system}", "{system} access", "Role change in {system}"),
            problems=("I need an account for {system}, my manager has approved it.",
                      "I can log in to {system} but I can't see the approvals screen anymore.",
                      "My {system} access was removed after my team change."),
            details=("I need the same access as {person}.", "The ticket from last month was closed without a fix."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Accounts for new starter {person}", "New user setup", "New joiner accounts"),
            problems=("{person} joins {team} on {day}; please create their network account and email.",
                      "Can you set up accounts for {person}, a contractor starting {day}?"),
            details=("Their line manager is {person}.", "They need the standard {team} access."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Leaver: disable account", "Leaver request for {person}", "Account removal for leaver"),
            problems=("{person} leaves on {day}, please disable their accounts at 5pm.",
                      "{person} is leaving the company, please remove their access after their last day."),
            details=("HR has confirmed the date.", "Please forward their email to their manager."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Contractor still has access", "Ex-contractor can still log in", "Old account still active"),
            problems=("A contractor's contract ended last week but they can still log in to {system}.",
                      "Someone who left last month still has an active account in {system}."),
            details=("HR confirmed their end date.",),
            priorities=("Medium", "High"),
        ),
        Issue(
            subjects=("Single sign-on not working", "Can't log in to {system}", "SSO error"),
            problems=("Single sign-on is failing with an 'AADSTS50011' error when we open {system}.",
                      "We can't log in to {system} this morning, it loops back to the login page."),
            details=("Password and MFA both work for email.", "It started around 8am."),
            priorities=("Medium", "High", "Critical"),
            scale="shared",
        ),
    ),
    "Email": (
        Issue(
            subjects=("Not receiving emails", "Emails not arriving", "Missing emails"),
            problems=("I haven't received any external emails {since}.",
                      "Emails from one of our clients aren't arriving, they say they sent them.",
                      "My inbox stopped updating, the last email I got was hours ago."),
            details=("Internal emails arrive fine.", "Nothing in the junk folder either."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Can't send email", "Problem sending emails", "Email sending issue"),
            problems=("Every email I send to external addresses bounces back with '550 5.7.1'.",
                      "My emails are stuck in the Outbox.",
                      "I get 'Mailbox full' when I try to send anything."),
            details=("The internet works fine.", "Sending from webmail works."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Outlook problem", "Outlook not working", "Issue with Outlook"),
            problems=("Outlook is stuck on 'Loading profile'.",
                      "I can browse the internet fine but Outlook says 'Trying to connect...'.",
                      "Outlook search doesn't find any emails older than a week."),
            details=("I restarted Outlook and my laptop.", "Webmail works fine."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Access to {dl}", "Request: add me to {dl}", "{dl} issue"),
            problems=("Please add me to {dl}.",
                      "{dl} disappeared from my Outlook folder list.",
                      "I need send-as permission for {dl}."),
            details=("My manager {person} approved this.", "Others in {team} already have it."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Calendar problem", "Calendar issue", "Outlook calendar not working right"),
            problems=("My calendar on my phone doesn't show the meetings I accept on my laptop.",
                      "Meeting invites I send show an hour off for people in other time zones.",
                      "I can't see {person}'s calendar anymore even though they shared it with me."),
            details=("I checked the time zone settings.", "It worked until last week."),
            priorities=LMH,
        ),
        Issue(
            subjects=("Mailbox almost full", "Mailbox size warning", "How to archive email"),
            problems=("I keep getting warnings that my mailbox is almost full.",
                      "How do I archive old emails? My mailbox is at 49.5 GB."),
            details=("I've already emptied deleted items.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Email on my phone", "Work email on mobile", "phone mail not syncing"),
            problems=("Email stopped syncing on my iPhone after I changed my password.",
                      "Can you help me set up work email on my new Android phone?"),
            details=("I'm using the Outlook app.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Email outage?", "Email problem", "Mail issue"),
            problems=("We haven't received any email for the last hour.",
                      "Our outgoing emails are being rejected as spam by customers' mail servers.",
                      "The invoices@ mailbox stopped receiving, and customers send payment confirmations there."),
            details=("Webmail shows the same.", "The bounce message mentions an SPF failure."),
            priorities=("High", "Critical"),
            scale="shared",
        ),
        Issue(
            subjects=("Attachment problem", "Issue with email attachments", "Attachment too large"),
            problems=("I can't send a 30 MB file, it says it exceeds the size limit.",
                      "PDF attachments arrive as winmail.dat for the people I send them to."),
            details=("It's a scanned contract.",),
            priorities=LM,
        ),
    ),
    "Security": (
        Issue(
            subjects=("Suspicious email", "Phishing?", "Is this email legit?", "Report phishing"),
            problems=("I received an email claiming to be from our CEO asking me to buy gift cards.",
                      "I got an email asking me to reset my password through a link that doesn't look like ours.",
                      "There's an email with an invoice attachment from a supplier we don't use."),
            details=("I didn't click anything.", "I've forwarded it to you as an attachment.",
                     "A colleague got the same one."),
            priorities=LMH,
        ),
        Issue(
            subjects=("I think I fell for a phishing email", "Possible compromise", "Clicked something I shouldn't have"),
            problems=("I clicked a link in an email and typed in my login details before I realised it was fake.",
                      "I opened an attachment from an unknown sender and now my laptop is acting strangely."),
            details=("I changed my password straight away.", "It was about 20 minutes ago."),
            priorities=("High", "Critical"),
            scale="incident",
        ),
        Issue(
            subjects=("Ransomware?", "Files encrypted", "Ransom note on my screen"),
            problems=("My files have all been renamed with a .locked extension and there's a ransom note on the desktop.",
                      "A message popped up saying my files are encrypted and I have to pay in bitcoin to get them back."),
            details=("I unplugged the network cable.", "I haven't restarted it."),
            priorities=("Critical",),
            scale="incident",
        ),
        Issue(
            subjects=("Virus warning", "Malware on my laptop", "Antivirus alert"),
            problems=("The antivirus says it found a trojan and couldn't remove it.",
                      "Defender keeps flagging a file called invoice_viewer.exe in my Downloads folder."),
            details=("I haven't opened the file.", "It came from a USB stick a visitor gave me."),
            priorities=("High", "Critical"),
            scale="incident",
        ),
        Issue(
            subjects=("Weird pop-ups", "Virus?", "Browser pop-ups saying I'm infected"),
            problems=("Pop-ups keep appearing saying my computer is infected and to call a phone number.",
                      "My browser home page changed to a strange search site and I keep getting ads."),
            details=("I didn't call the number.", "I closed the browser."),
            priorities=("Medium", "High"),
        ),
        Issue(
            subjects=("Lost/stolen device", "Lost work device", "Device missing"),
            problems=("My work laptop was stolen from my car last night.",
                      "I left my company phone in a taxi.",
                      "I think I lost my laptop bag at the airport."),
            details=("I've reported it to the police.", "It was locked with a PIN."),
            priorities=("High", "Critical"),
            scale="incident",
        ),
        Issue(
            subjects=("Suspicious login alert", "Unknown sign-in", "Someone in my account?"),
            problems=("I got an alert about a sign-in to my account from another country.",
                      "There are emails in my Sent folder that I didn't write.",
                      "My MFA app keeps asking me to approve logins I didn't make."),
            details=("I haven't approved any of them.", "I'm not travelling."),
            priorities=("High", "Critical"),
            scale="incident",
        ),
        Issue(
            subjects=("Sent data to the wrong person", "Sensitive data shared by mistake", "Data leak?"),
            problems=("I accidentally sent a spreadsheet with employee salaries to an external address.",
                      "A file with customer data was shared publicly via a link."),
            details=("I tried to recall the message.", "I don't know who has opened it."),
            priorities=("High", "Critical"),
            scale="incident",
        ),
        Issue(
            subjects=("Security policy question", "Is this allowed?", "Question for the security team"),
            problems=("Is it okay to use a personal USB stick to move files between my laptops?",
                      "Can I install a password manager on my work laptop?",
                      "What's the policy on pasting company documents into online AI tools?"),
            details=("Just want to check before I do it.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Physical security concern", "Security concern in the office", "Report: secure door"),
            problems=("Someone I didn't recognise followed me through the secure door this morning.",
                      "The server room door on the {floor} has been propped open all day."),
            details=("They didn't have a visible badge.",),
            priorities=("Medium", "High"),
        ),
    ),
    "Printer": (
        Issue(
            subjects=("Printer jam", "Paper jam", "Printer jammed again"),
            problems=("{printer} keeps saying paper jam but there's no paper stuck.",
                      "{printer} is jammed in tray 2 and I can't get the paper out.",
                      "{printer} jams on every double-sided job."),
            details=("I opened all the doors and checked.", "It's been like this {since}."),
            priorities=LMH,
            scale="shared",
        ),
        Issue(
            subjects=("Can't print", "Print jobs stuck", "Printer offline"),
            problems=("My print jobs go to the queue but nothing comes out of {printer}.",
                      "{printer} shows as offline on my laptop.",
                      "Jobs to {printer} just sit in the queue."),
            details=("I restarted the print spooler.", "I can print to other printers."),
            priorities=LMH,
            scale="shared",
        ),
        Issue(
            subjects=("Printer setup", "Can't add printer", "Printer driver"),
            problems=("How do I add {printer} to my new laptop?",
                      "{printer} doesn't show up in the list when I try to add it.",
                      "Printing to {printer} from my Mac gives pages of garbled characters."),
            details=("I followed the intranet guide.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Print quality issue", "Bad print quality", "Printouts look wrong"),
            problems=("Everything printed from {printer} has a grey streak down the middle.",
                      "Prints from {printer} are very faded, I think the toner is low.",
                      "Colours are completely wrong on {printer}."),
            details=("I ran the cleaning cycle.",),
            priorities=LM,
            scale="shared",
        ),
        Issue(
            subjects=("Scan to email not working", "Scanner not sending", "Can't scan"),
            problems=("When I scan on {printer} the email never arrives in my inbox.",
                      "Scan to folder on {printer} fails with 'Cannot connect to server'.",
                      "The scanner on {printer} shows 'SMTP error' after scanning."),
            details=("Normal emails arrive fine.", "It worked last week."),
            priorities=LMH,
            scale="shared",
        ),
        Issue(
            subjects=("Toner replacement", "Printer out of toner", "Need toner"),
            problems=("{printer} is out of black toner.",
                      "Can we get more toner cartridges for {printer}?"),
            details=("The display says 'Replace cartridge'.",),
            priorities=LM,
            scale="shared",
        ),
        Issue(
            subjects=("Label printer down", "Can't print shipping labels", "Dispatch printer not working"),
            problems=("The warehouse label printer stopped printing and we can't ship orders.",
                      "The shipping label printer at the dispatch desk only prints blank labels."),
            details=("We've reloaded the label roll.", "The courier collects at {time}."),
            priorities=("High", "Critical"),
            scale="shared",
        ),
    ),
    "Other": (
        Issue(
            subjects=("Air conditioning not working", "Too hot on the {floor}", "Heating/cooling problem"),
            problems=("The air conditioning on the {floor} isn't working and it's very hot.",
                      "The heating is off on the {floor} and everyone is wearing coats."),
            details=("I've asked reception but they said to raise a ticket.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Broken chair", "Need a new chair", "Desk chair broken"),
            problems=("My chair is broken, the height adjustment doesn't work.",
                      "Could I get an ergonomic chair? I have a note from my doctor."),
            priorities=LM,
        ),
        Issue(
            subjects=("Desk move request", "Can I change desks?", "Seating request"),
            problems=("Can I move desks to sit with the {team} team?",
                      "We need four desks next to each other for a project team starting {day}."),
            priorities=LM,
        ),
        Issue(
            subjects=("Question", "Who do I contact?", "General enquiry", "Quick question"),
            problems=("Who should I contact about a payroll question?",
                      "Is there an induction session for new starters this month?",
                      "Where can I find the holiday request form?"),
            details=("Sorry if this is the wrong place.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Onboarding schedule", "First day for {person}", "Induction plan"),
            problems=("What time should {person} arrive on their first day and who will meet them?",
                      "Can someone show the new interns around the {site} office on {day}?"),
            details=("They start in {team}.",),
            priorities=LM,
        ),
        Issue(
            subjects=("Lost building badge", "Replacement badge", "Badge request"),
            problems=("I lost my building badge, how do I get a new one?",
                      "My door badge doesn't open the {floor} door anymore."),
            details=("Reception sent me here.",),
            priorities=LMH,
        ),
        Issue(
            subjects=("Parking permit", "Parking request", "Car park access"),
            problems=("Can I get a parking permit for the {site} office?",
                      "The car park barrier doesn't recognise my registration anymore."),
            priorities=LM,
        ),
        Issue(
            subjects=("Thank you", "Feedback", "Thanks for the help"),
            problems=("Just wanted to say thanks to {person} for fixing my laptop so quickly.",
                      "Is there a way to give feedback on the new intranet design?"),
            priorities=("Low",),
        ),
        Issue(
            subjects=("Office supplies", "Stationery / business cards", "Order request"),
            problems=("How do I order new business cards?",
                      "We need more notebooks and pens for the training room."),
            priorities=LM,
        ),
        Issue(
            subjects=("Kitchen issue", "Problem in the {floor} kitchen", "Kitchen equipment broken"),
            problems=("The coffee machine on the {floor} shows an error and won't make coffee.",
                      "The fridge in the {floor} kitchen is leaking water onto the floor."),
            priorities=LM,
        ),
        Issue(
            subjects=("Building issue", "Facilities problem", "Urgent facilities issue"),
            problems=("Water is dripping from the ceiling above our desks on the {floor}.",
                      "The lift is broken and a colleague who uses a wheelchair can't get to the {floor}."),
            details=("Facilities' phone line isn't answering.",),
            priorities=("Medium", "High"),
        ),
    ),
}

# --------------------------------------------------------------------------- priority factors
# Probability of each factor value given the priority label.
PRIORITY_FACTORS: dict[str, dict[str, dict[str, float]]] = {
    "Low": {
        "blocking": {"none": 0.8, "slowed": 0.2},
        "deadline": {"none": 0.85, "soon": 0.15},
    },
    "Medium": {
        "blocking": {"slowed": 0.8, "none": 0.2},
        "deadline": {"soon": 0.6, "none": 0.35, "today": 0.05},
    },
    "High": {
        "blocking": {"blocked": 0.85, "slowed": 0.15},
        "deadline": {"today": 0.55, "none": 0.3, "now": 0.15},
    },
    "Critical": {
        "blocking": {"blocked": 1.0},
        "deadline": {"now": 0.7, "today": 0.2, "none": 0.1},
    },
}

# Scope depends on whether the issue is personal, a shared service, or a security incident.
SCOPE_BY_SCALE: dict[str, dict[str, dict[str, float]]] = {
    "single": {
        "Low": {"self": 1.0},
        "Medium": {"self": 1.0},
        "High": {"self": 1.0},
        "Critical": {"vip": 1.0},
    },
    "shared": {
        "Low": {"self": 0.8, "team": 0.2},
        "Medium": {"self": 0.5, "team": 0.5},
        "High": {"team": 0.8, "self": 0.2},
        "Critical": {"org": 0.6, "customers": 0.4},
    },
    "incident": {
        "Low": {"self": 1.0},
        "Medium": {"self": 1.0},
        "High": {"self": 0.8, "team": 0.2},
        "Critical": {"incident_wide": 1.0},
    },
}

# Phrases must make sense for ANY issue (a badge, a printer, a password...).
FACTOR_PHRASES: dict[str, dict[str, tuple[str, ...]]] = {
    "scope": {
        "self": ("It's only affecting me as far as I know.", "Only me so far.", "Nobody else has mentioned it.",
                 "Just me, my colleagues are fine."),
        "team": ("{n} people in {team} have the same problem.", "The whole {team} team is affected.",
                 "It's happening to everyone in my team.", "A few of us on the {floor} are affected.",
                 "Everyone sitting near me has the same issue."),
        "org": ("It's affecting the entire {site} office.", "Nobody in the company can work.",
                "This is happening to everyone in the building.", "All {big_n} people at {site} are affected.",
                "Every department is reporting it."),
        "customers": ("Customers can't {customer_action}.", "This is customer-facing: clients can't {customer_action}.",
                      "Our support line is down and customers can't reach us.",
                      "We're losing sales every minute this is down."),
        "vip": ("This is for our CEO, who is presenting to investors at {time}.",
                "It's for the CFO, and the board meeting starts in an hour.",
                "I'm about to go live on a customer webinar in 20 minutes.",
                "Our managing director needs this sorted before flying out today.",
                "I'm on stage at the client conference in 30 minutes and need this."),
        "incident_wide": ("Files on {shared_drive} are being changed too.", "I have admin rights on {system}.",
                          "The device has all our customer records on it.",
                          "Several people got the same email and some of them clicked it.",
                          "It looks like it's spreading to other machines."),
    },
    "blocking": {
        "none": ("I have a workaround for now.", "It's not stopping me from working.",
                 "It's more of a request than a problem.", "It's not a big deal.",
                 "I can manage without it for now."),
        "slowed": ("It's slowing me down quite a bit.", "I can still work but it's frustrating.",
                   "It keeps interrupting my work.", "I'm losing time every day because of this.",
                   "It's making everything take twice as long."),
        "blocked": ("I can't do any work at all.", "I'm completely stuck.", "I can't do my job until this is fixed.",
                    "I'm totally blocked.", "We can't work at all."),
    },
    "deadline": {
        "none": ("No rush.", "Whenever you get a chance.", "Not urgent.", "Whenever is convenient."),
        "soon": ("Would be good to get this sorted this week.", "In the next few days if possible.",
                 "Could someone look at it before {day}?", "Sometime this week would be great."),
        "today": ("I need this sorted before my {time} meeting.", "We have a deadline at {time} today.",
                  "Please could it be fixed today?", "I'm presenting to a client this afternoon.",
                  "Payroll has to be submitted by end of day."),
        "now": ("Please treat this as an emergency.", "We need help right now.", "This needs immediate attention.",
                "Please call me ASAP on {phone}.", "Every minute counts here."),
    },
}

# Chance that a factor is left implicit (not written down), per factor value.
OMIT_PROBABILITY: dict[str, float] = {"self": 0.5, "none": 0.4, "slowed": 0.15, "soon": 0.2, "default": 0.1}

# --------------------------------------------------------------------------- writing styles
STYLES: dict[str, dict[str, tuple[str, ...]]] = {
    "formal": {
        "greetings": ("Dear IT Support,", "Hello IT team,", "Good morning,", "Dear Service Desk,"),
        "closings": ("Kind regards,\n{person}", "Thank you in advance.\n{person}", "Best regards,\n{person} ({team})",
                     "Many thanks,\n{person}"),
    },
    "casual": {
        "greetings": ("Hi!", "hey team", "Hi IT,", "Hiya,", "Morning all,", ""),
        "closings": ("thanks!", "cheers, {person}", "thx", "Thanks a lot", ""),
    },
    "terse": {"greetings": ("",), "closings": ("",)},
    "frustrated": {
        "greetings": ("", "Hi,", "IT,"),
        "closings": ("", "Thanks.", "{person}"),
        "extras": ("This is the third time I've reported this.", "Honestly this keeps happening.",
                   "Please actually fix it this time.", "I've been waiting two days for a reply already."),
    },
    "polite_nonnative": {
        "greetings": ("Dear Sir/Madam,", "Hello team,", "Respected IT team,"),
        "closings": ("Kindly do the needful.", "Kindly help on priority.", "Please revert at the earliest.",
                     "Thanks and regards,\n{person}"),
    },
    "mobile": {
        "greetings": ("", "Hi"),
        "closings": ("Sent from my iPhone", "Sent from Outlook for Android", "Sent from my phone, sorry for typos"),
    },
}
STYLE_WEIGHTS: dict[str, float] = {
    "formal": 0.25, "casual": 0.25, "terse": 0.15, "frustrated": 0.1, "polite_nonnative": 0.12, "mobile": 0.13,
}

# Background sentences that add realism (and cross-category words) without changing the label.
CONTEXT_SENTENCES: tuple[str, ...] = (
    "I'm working from home today.", "I'm in the {site} office this week.", "I already restarted my laptop.",
    "I tried logging out and back in.", "I'm on a Mac.", "This is my first week so sorry if it's a silly question.",
    "My manager asked me to raise a ticket.", "Ticket number from last time was INC0{n}482.",
)

# Side requests from ANOTHER category appended to some tickets ("Also, ..."). The label stays the
# main issue, so keyword counting is misled but a reader is not.
SIDE_REQUEST_PREFIXES: tuple[str, ...] = ("Also, ", "Unrelated, but ", "PS: ", "While I have you, ", "Separately, ")
SIDE_REQUESTS: dict[str, tuple[str, ...]] = {
    "Hardware": ("my mouse is a bit worn out, could I get a new one at some point?",
                 "the battery on my laptop isn't great anymore, no rush on that one."),
    "Software": ("could someone install Notepad++ for me when you have time?",
                 "Excel has been a bit slow lately, not a big deal."),
    "Network": ("the Wi-Fi in the kitchen is a bit weak.",
                "the VPN was slow last week but it's fine now."),
    "Access/Accounts": ("I still can't see the HR SharePoint site, but that can wait.",
                        "my password expires next week, will I get a reminder?"),
    "Email": ("could you add me to the all-staff list?",
              "my mailbox keeps saying it's almost full."),
    "Security": ("I got a weird email yesterday too, I just deleted it.",
                 "is it okay to use my personal USB stick for presentations?"),
    "Printer": ("the printer by the kitchen is low on toner.",
                "the scanner on our floor sometimes takes ages to send."),
    "Other": ("the coffee machine is broken again!",
              "who do I ask about a parking permit?"),
}

# Light surface variation applied after rendering (each pair is swapped with a small probability).
SYNONYMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("won't", ("will not", "doesn't")),
    ("can't", ("cannot",)),
    ("isn't working", ("is not working", "has stopped working", "is broken")),
    ("laptop", ("machine", "notebook", "computer")),
    ("really", ("very", "super")),
    ("Could you", ("Can you", "Would you be able to")),
    ("please", ("pls", "kindly")),
)

VAGUE_SUBJECTS: tuple[str, ...] = ("Help", "Issue", "problem", "Quick question", "IT issue", "Not working",
                                   "Request", "(no subject)", "help please")
# Added to some urgent tickets AND to some routine ones (people over-flag), so it is only a weak cue.
URGENCY_PREFIXES: tuple[str, ...] = ("URGENT: ", "URGENT - ", "!! ", "ASAP: ", "High priority: ")
