import click
from src.logic import Stack


@click.command()
@click.option(
    "--location",
    help="This specifies the location you want to know the time. For example, Lagos or London",
)
@click.option(
    "--zone",
    help="The timezone information you need. Ensure it is properly capitalized, for example CET or WAT",
)
def main(location, zone):
    st = Stack()
    if location:
        st.push(location)
    if zone:
        st.pop(zone)



if __name__ == "__main__":
    main()